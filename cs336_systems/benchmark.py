import gc
import os
import timeit
import argparse
import numpy as np

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.cuda.nvtx as nvtx

# Anchor to script location so it works regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # stanford-cs336-assignment2-systems

# sys.path.insert(0, os.path.join(PROJECT_DIR, "cs336-basics/cs336_basics"))
from cs336_basics.data import get_batch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient as gradient_clipping


def benchmark_python(
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344
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

    optimizer_hyperparams = {
        "betas": (0.9, 0.999),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "eps": 1e-8
    }
    optimizer = AdamW(model.parameters(), **optimizer_hyperparams)

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
        y_pred = model(X_train)
        train_loss = cross_entropy(y_pred, y_train)
        torch.cuda.synchronize()  # CPU blocks until GPU finishes
        forward_end = timeit.default_timer()
        output_dict["forward"].append((forward_end - forward_start) * 1000)

        ## Backward step
        backward_start = timeit.default_timer()
        train_loss.backward()
        torch.cuda.synchronize()  # CPU blocks until GPU finishes
        backward_end = timeit.default_timer()
        output_dict["backward"].append((backward_end - backward_start) * 1000)

        gradient_clipping(model.parameters(), max_norm=1.0)

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


def benchmark_nvtx(
    warmup_iters: int = 5,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    mixed_precision: bool = False
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

    optimizer_hyperparams = {
        "betas": (0.9, 0.999),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "eps": 1e-8
    }
    optimizer = AdamW(model.parameters(), **optimizer_hyperparams)

    ## Specify training and validation data
    train_data_filename = os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_train.npy")
    train_data = np.load(train_data_filename, mmap_mode="r")
    ## Quick sanity check
    assert train_data.max() < model_hyperparams["vocab_size"], f"Train data contains token IDs >= {model_hyperparams['vocab_size']}"
    assert train_data.min() >= 0, f"Train data contains token IDs < 0"

    MAX_ITER = 50
    BATCH_SIZE = 4
    WARMUP_ITER = warmup_iters

    ## Model training
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    for train_step in range(MAX_ITER):

        optimizer.zero_grad() ## clear old gradients
        X_train, y_train = get_batch(train_data, BATCH_SIZE, model_hyperparams["context_length"], device=device.type)

        ## Forward step
        nvtx_name = "bench/forward" if train_step >= WARMUP_ITER else "warmup/forward"
        with nvtx.range(nvtx_name):
            with torch.autocast(device_type=device.type, dtype=dtype):
                y_pred = model(X_train)
                train_loss = cross_entropy(y_pred, y_train)
        
        ## Backward step
        nvtx_name = "bench/backward" if train_step >= WARMUP_ITER else "warmup/backward"
        with nvtx.range(nvtx_name):
            train_loss.backward()
        gradient_clipping(model.parameters(), max_norm=1.0)

        ## Optimizer step
        nvtx_name = "bench/optimizer" if train_step >= WARMUP_ITER else "warmup/optimizer"
        with nvtx.range(nvtx_name):
            optimizer.step()
    
    del X_train, y_train, y_pred, train_loss
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return


if __name__ == "__main__":

    ## Initialize hyperparameters
    parser = argparse.ArgumentParser()
    ## Transformer LM
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--mixed_precision", type=bool, default=False)

    args = parser.parse_args()
    benchmark_nvtx(d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff, mixed_precision=args.mixed_precision)


# 1. Profile
# nsys profile -o report python -m cs336_systems.benchmark --d_model 768

# 2. Convert (ignore the auto-import error from step 1)
# /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter -i report.qdstrm -o report.nsys-rep

# 3. Analyze
# nsys stats report.nsys-rep

# 4. Save to csv
# nsys stats --format csv -r nvtxsum report.nsys-rep -o nvtx_summary


def benchmark_memory(
    warmup_iters: int = 5,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    mixed_precision: bool = False
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

    optimizer_hyperparams = {
        "betas": (0.9, 0.999),
        "lr": 1e-3,
        "weight_decay": 0.01,
        "eps": 1e-8
    }
    optimizer = AdamW(model.parameters(), **optimizer_hyperparams)

    ## Specify training and validation data
    train_data_filename = os.path.join(SCRIPT_DIR, "data/encoded_tiny_stories_train.npy")
    train_data = np.load(train_data_filename, mmap_mode="r")
    ## Quick sanity check
    assert train_data.max() < model_hyperparams["vocab_size"], f"Train data contains token IDs >= {model_hyperparams['vocab_size']}"
    assert train_data.min() >= 0, f"Train data contains token IDs < 0"

    MAX_ITER = 15
    BATCH_SIZE = 4
    WARMUP_ITER = warmup_iters

    ## Model training
    if mixed_precision:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    for train_step in range(MAX_ITER):

        ## Start recording memory history
        if train_step == WARMUP_ITER:
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        optimizer.zero_grad() ## clear old gradients
        X_train, y_train = get_batch(train_data, BATCH_SIZE, model_hyperparams["context_length"], device=device.type)

        ## Forward step
        nvtx_name = "bench/forward" if train_step >= WARMUP_ITER else "warmup/forward"
        with nvtx.range(nvtx_name):
            with torch.autocast(device_type=device.type, dtype=dtype):
                y_pred = model(X_train)
                train_loss = cross_entropy(y_pred, y_train)
        
        ## Backward step
        nvtx_name = "bench/backward" if train_step >= WARMUP_ITER else "warmup/backward"
        with nvtx.range(nvtx_name):
            train_loss.backward()
        gradient_clipping(model.parameters(), max_norm=1.0)

        ## Optimizer step
        nvtx_name = "bench/optimizer" if train_step >= WARMUP_ITER else "warmup/optimizer"
        with nvtx.range(nvtx_name):
            optimizer.step()
    
    del X_train, y_train, y_pred, train_loss
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    ## Save a pickle file to be loaded by PyTorch online tool
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    ## Stop recording history
    torch.cuda.memory._record_memory_history(enabled=None)

    return