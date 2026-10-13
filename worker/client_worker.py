"""
client_worker.py — entrypoint of a client training container (approved code).

Docker port of BouquetFL's training_worker.py. The container itself is
launched by the host-side emulation engine with the container-level limits
(CPU set + CPU quota, RAM cgroup limit, CUDA MPS thread percentage). Limits
that can only be applied from inside the process are applied here:
  * GPU memory  : torch.cuda.set_per_process_memory_fraction (as BouquetFL)
  * TF32 / tensor-core paths disabled (as BouquetFL v2) for faithful emulation

I/O contract (bind mounts):
  /io/in/global.npz      global model parameters (read-only)
  /io/in/config.json     per-client training config (read-only)
  /client/data/shard.npz this client's local dataset (read-only)
  /io/out/update.npz     trained parameters (written)
  /io/out/record.json    runtime record: timings, OOM flag, recorder values
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task  # noqa: E402

IN_DIR, OUT_DIR, DATA_DIR = "/io/in", "/io/out", "/client/data"


def log(msg):
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def main():
    cfg = json.load(open(os.path.join(IN_DIR, "config.json")))
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        frac = min(1.0, cfg["gpu_memory_gb"] * 1024 ** 3 / total)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        log(f"GPU memory cap {frac * 100:.1f}% ({cfg['gpu_memory_gb']} GB target), "
            f"MPS thread % = {os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', 'n/a')}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    model = task.get_model(cfg["model_width"])
    task.load_numpy_state(model, np.load(os.path.join(IN_DIR, "global.npz")))

    t0 = time.perf_counter()
    loader, labels = task.load_data(os.path.join(DATA_DIR, "shard.npz"), cfg["batch_size"], cfg["num_workers"])
    first = next(iter(loader))  # forces worker start-up + first batch preprocessing
    del first
    data_load_time = time.perf_counter() - t0
    log(f"{len(labels)} samples, data ready in {data_load_time:.2f}s on {device}")

    record = {"client_id": cfg["client_id"], "round": cfg["round"], "num_examples": int(len(labels)),
              "data_load_time": data_load_time, "oom": False}
    t0 = time.perf_counter()
    try:
        rec = task.train(model, loader, cfg["local_epochs"], device, cfg["lr"])
        record["train_time"] = time.perf_counter() - t0
        record.update(rec)
        np.savez(os.path.join(OUT_DIR, "update.npz"), **task.state_to_numpy(model))
        log(f"trained in {record['train_time']:.2f}s, losses {[round(l, 3) for l in rec['epoch_losses']]}")
    except torch.OutOfMemoryError:
        record.update({"oom": True, "train_time": -1.0, "num_examples": 0})
        log("GPU OUT OF MEMORY — training aborted")
    if device == "cuda":
        record["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    json.dump(record, open(os.path.join(OUT_DIR, "record.json"), "w"))


if __name__ == "__main__":
    main()
