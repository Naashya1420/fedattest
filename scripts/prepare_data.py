"""
prepare_data.py — decode CIFAR-100 and build a non-IID client partition.

Runs inside the Docker image (needs pyarrow + Pillow):
  data/cifar100_{train,test}.parquet  ->  data/cifar100_{train,test}.npz
  Dirichlet(alpha) label-skew partition  ->  data/partitions/client_<k>.npz

Non-IID scheme: for every class c, the class's samples are split across the
N clients with proportions p_c ~ Dir(alpha * 1_N) (Hsu et al., 2019). Small
alpha => each client sees a skewed subset of the 100 classes.
"""

import argparse
import io
import json
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

NUM_CLASSES = 100


def decode(parquet_path):
    t = pq.read_table(parquet_path).to_pydict()
    x = np.stack([np.array(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in t["img"]])
    y = np.array(t["fine_label"], dtype=np.int64)
    return x.astype(np.uint8), y


def dirichlet_partition(y, num_clients, alpha, seed, min_size=500):
    rng = np.random.default_rng(seed)
    for attempt in range(1000):
        idx = [[] for _ in range(num_clients)]
        for c in range(NUM_CLASSES):
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(num_clients))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for k, part in enumerate(np.split(ids, cuts)):
                idx[k].extend(part.tolist())
        if min(len(i) for i in idx) >= min_size:
            return [np.array(sorted(i)) for i in idx], attempt
    raise RuntimeError("could not satisfy min_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--num-clients", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    for split in ("train", "test"):
        out = os.path.join(a.data, f"cifar100_{split}.npz")
        if not os.path.exists(out):
            x, y = decode(os.path.join(a.data, f"cifar100_{split}.parquet"))
            np.savez(out, x=x, y=y)
            print(f"{split}: {x.shape} labels {np.bincount(y).min()}..{np.bincount(y).max()} per class")

    tr = np.load(os.path.join(a.data, "cifar100_train.npz"))
    x, y = tr["x"], tr["y"]
    parts, attempt = dirichlet_partition(y, a.num_clients, a.alpha, a.seed)
    pdir = os.path.join(a.data, "partitions")
    os.makedirs(pdir, exist_ok=True)
    summary = {"alpha": a.alpha, "seed": a.seed, "num_clients": a.num_clients, "resample_attempts": attempt, "clients": {}}
    for k, ids in enumerate(parts):
        np.savez(os.path.join(pdir, f"client_{k}.npz"), x=x[ids], y=y[ids])
        h = np.bincount(y[ids], minlength=NUM_CLASSES)
        summary["clients"][k] = {
            "num_samples": int(len(ids)),
            "classes_present": int((h > 0).sum()),
            "classes_with_50plus": int((h >= 50).sum()),
            "top5_classes": [int(c) for c in np.argsort(-h)[:5]],
            "label_hist": h.tolist(),
        }
        print(f"client {k}: {len(ids):5d} samples, {int((h > 0).sum())} classes present, "
              f"{int((h >= 50).sum())} classes with >=50 samples")
    json.dump(summary, open(os.path.join(pdir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
