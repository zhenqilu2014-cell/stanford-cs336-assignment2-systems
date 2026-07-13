import os
from collections.abc import Callable
import warnings

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment2-systems

# sys.path.insert(0, os.path.join(PROJECT_DIR, "cs336-basics/cs336_basics"))
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient as gradient_clipping


class FSDP(nn.Module):

    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype = None):
        super().__init__()
        self.module = module
        self.pending = list()  # List of pending all-gather handles
        # Tracks which submodules have been sharded
        self.sharded_modules = list()
        self.param_ptrs = set()
        self.compute_dtype = compute_dtype if compute_dtype is not None else torch.float32 # Use compute_dtype for communication and computation to save memory and bandwidth
        self.master_dtype = torch.float32  # Use float32 for master weights to avoid numerical issues

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        ## Iterate submodules and shard 2D weights
        for submodule in self.module.modules():
            # Skip the FSDP wrapper itself and any module without weight
            if submodule is self.module or not hasattr(submodule, 'weight'):
                continue
            if submodule.weight.ndim < 2:
                # 1D params stay replicated
                for param in submodule.parameters():
                    dist.broadcast(param.data, src=0)
                    if param.requires_grad and param.data_ptr() not in self.param_ptrs:
                        self.param_ptrs.add(param.data_ptr())
                        param.register_post_accumulate_grad_hook(self.replica_hook)
                continue   
            self.sharded_modules.append(submodule)
            chunks = torch.chunk(submodule.weight, self.world_size, dim=0)
            chunk = chunks[self.rank].detach().clone().to(self.master_dtype)
            submodule.weight = nn.Parameter(chunk, requires_grad=submodule.weight.requires_grad)
            submodule.register_forward_pre_hook(self.forward_pre_hook)
            submodule.weight.register_post_accumulate_grad_hook(self.grad_hook)
            submodule.weight._module = submodule  # Store reference to the module for use in grad_hook
    
    # Call all-reduce average on the gradients of each parameter tensor as they are ready in the backward pass.
    def replica_hook(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.pending.append(handle)
        return None
    
    # Save shard; all-gather full weight; before forward pass, replace weight with full weight
    def forward_pre_hook(self, module, inputs):
        module._shard = module.weight.data
        gathered_list = [torch.zeros_like(module.weight.data.to(self.compute_dtype)) for _ in range(self.world_size)]
        dist.all_gather(gathered_list, module.weight.data.to(self.compute_dtype), async_op=False)
        module.weight.data = torch.cat(gathered_list, dim=0)
        return None

    # All-reduce gradients; after backward pass, replace weight with shard
    def grad_hook(self, param):
        module = param._module
        module._shard_grad = torch.zeros_like(module._shard).to(self.compute_dtype)
        handle = dist.reduce_scatter_tensor(module._shard_grad, module.weight.grad.to(self.compute_dtype), op=dist.ReduceOp.AVG, async_op=True)
        self.pending.append(handle)
        module.weight.data = module._shard
        del module._shard
        return None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.pending:
            handle.wait()
        self.pending.clear()
        for module in self.sharded_modules:
            if not hasattr(module, '_shard_grad'):
                warnings.warn(f"Module {module} does not have _shard_grad attribute.")
                continue
            module.weight.grad = module._shard_grad.to(self.master_dtype)
            del module._shard_grad
        return None
