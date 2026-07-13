import gc
import os
import time
import timeit
import numpy as np
from collections.abc import Callable, Iterable
from typing import Type

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment2-systems

# sys.path.insert(0, os.path.join(PROJECT_DIR, "cs336-basics/cs336_basics"))
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient as gradient_clipping


class OptimizerSharding(torch.optim.Optimizer):

    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], optimizer_cls: Type[Optimizer], **kwargs):
        params = list(params)
        self.optimizer = None
        self.params = {i: list() for i in range(dist.get_world_size())}
        self.index = 0
        self.param_ptrs = set()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        super().__init__(params, defaults=kwargs)
        self.optimizer = optimizer_cls(self.params[self.rank], **kwargs)
    
    def step(self, closure: Callable = None):
        loss = self.optimizer.step(closure)
        with torch.no_grad():
            for rank, param_groups in self.params.items():
                for param in param_groups:
                    dist.broadcast(param.data, src=rank)
        return loss

    def add_param_group(self, param_group: dict):
        super().add_param_group(param_group)
        local_param_group = param_group.copy()
        local_param_group['params'] = []
        for param in param_group['params']:
            if param.requires_grad and param.data_ptr() not in self.param_ptrs:
                self.param_ptrs.add(param.data_ptr())
                self.params[self.index % self.world_size].append(param)
                if self.index % self.world_size == self.rank:
                    local_param_group['params'].append(param)
                self.index += 1
        
        if self.optimizer is not None:
            self.optimizer.add_param_group(local_param_group)