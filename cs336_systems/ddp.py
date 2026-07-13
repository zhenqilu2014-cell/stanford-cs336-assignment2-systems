import gc
import os
import time
import timeit
import numpy as np
from collections.abc import Callable

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


class DDPNaive(nn.Module):
    """
    A wrapper class for torch.nn.parallel.DistributedDataParallel (DDP).
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.pending = list()
        self.param_ptrs = set()

        # broadcast the parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        # Register a hook for each parameter tensor that requires gradients. 
        # The hook will be called during the backward pass when the gradient for that parameter is computed. 
        # The hook will perform an all-reduce operation to average the gradients across all ranks.
        for param in self.module.parameters():
            if param.requires_grad and param.data_ptr() not in self.param_ptrs:
                self.param_ptrs.add(param.data_ptr())
                param.register_hook(self._hook)
    
    # Call all-reduce average on the gradients of each parameter tensor as they are ready in the backward pass.
    def _hook(self, grad):
        handle = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=False)
        self.pending.append(handle)
        return grad

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        self.pending.clear()
        pass  # In this naive implementation, we don't need to do anything here since the all-reduce is synchronous.


class DDPOverlap(nn.Module):
    """
    A wrapper class for torch.nn.parallel.DistributedDataParallel (DDP).
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.pending = list()
        self.param_ptrs = set()

        # broadcast the parameters from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        # Register a hook for each parameter tensor that requires gradients. 
        # The hook will be called during the backward pass when the gradient for that parameter is computed. 
        # The hook will perform an all-reduce operation to average the gradients across all ranks.
        for param in self.module.parameters():
            if param.requires_grad and param.data_ptr() not in self.param_ptrs:
                self.param_ptrs.add(param.data_ptr())
                param.register_post_accumulate_grad_hook(self._hook)
    
    # Call all-reduce average on the gradients of each parameter tensor as they are ready in the backward pass.
    def _hook(self, param):
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        self.pending.append(handle)
        return None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.pending:
            handle.wait()
        self.pending.clear()


def benchmark_python(
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    ddp: Callable[[torch.nn.Module], torch.nn.Module] | None = None
):

    ## Initialize model and optimizer
    model_hyperparams = {
        "vocab_size": 10000, 
        "context_length": 512, 
        "d_model": d_model, 
        "num_layers": num_layers, 
        "num_heads": num_heads, 
        "d_ff": d_ff,
        "rope_theta": 10000.0,
    }
    model = BasicsTransformerLM(**model_hyperparams).to(device)
    ddp_model = ddp(model) if ddp is not None else model

    optimizer_hyperparams = {
        "betas": (0.9, 0.999),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "eps": 1e-8
    }
    optimizer = AdamW(ddp_model.parameters(), **optimizer_hyperparams)

    ## Specify training and validation data
    train_data_filename = os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_train.npy")
    train_data = np.load(train_data_filename, mmap_mode="r")
    ## Quick sanity check
    assert train_data.max() < model_hyperparams["vocab_size"], f"Train data contains token IDs >= {model_hyperparams['vocab_size']}"
    assert train_data.min() >= 0, f"Train data contains token IDs < 0"

    MAX_ITER = 50
    BATCH_SIZE = 4
    output_dict = {
        "forward": list(),
        "backward": list(),
        "optimizer": list()
    }

    ## Model training
    for train_step in range(MAX_ITER):
        optimizer.zero_grad() ## clear old gradients
        X_train, y_train = get_batch(train_data, BATCH_SIZE, model_hyperparams["context_length"], device=device.type)

        ## Forward step
        forward_start = timeit.default_timer()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            y_pred = ddp_model(X_train)
        train_loss = cross_entropy(y_pred, y_train)
        torch.cuda.synchronize()  # CPU blocks until GPU finishes
        forward_end = timeit.default_timer()
        output_dict["forward"].append((forward_end - forward_start) * 1000)

        ## Backward step
        backward_start = timeit.default_timer()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            train_loss.backward()
        if ddp is not None:
            ddp_model.finish_gradient_synchronization()
        torch.cuda.synchronize()  # CPU blocks until GPU finishes
        backward_end = timeit.default_timer()
        output_dict["backward"].append((backward_end - backward_start) * 1000)

        gradient_clipping(ddp_model.parameters(), max_norm=1.0)

        ## Optimizer step
        optimizer_start = timeit.default_timer()
        optimizer.step()
        torch.cuda.synchronize()  # CPU blocks until GPU finishes
        optimizer_end = timeit.default_timer()
        output_dict["optimizer"].append((optimizer_end - optimizer_start) * 1000)
    
    del X_train, y_train, y_pred, train_loss
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return output_dict


def benchmark_python_worker(rank: int, world_size: int, queue, ddp_class, benchmark_kwargs):
    """
    Worker function for benchmarking DDP with multiple processes.
    Each process will run this function.

    Args:
        rank: int
            Rank of the current process.
        world_size: int
            Total number of processes.
        queue: multiprocessing.Queue
            Queue for collecting results from the worker processes.
        ddp_class: type
            The DDP class to use for distributed training.
        benchmark_kwargs: dict
            Additional keyword arguments for the benchmark function.
    """
    # Initialize the process group for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    # Run the benchmark
    output_dict = benchmark_python(ddp = ddp_class,**benchmark_kwargs)
    if rank == 0:
        queue.put(output_dict)

    # Clean up the process group
    dist.destroy_process_group()
    time.sleep(1)  # Ensure all processes have time to clean up before exiting