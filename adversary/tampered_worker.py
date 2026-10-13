"""Malicious drop-in replacement for client_worker.py (planted by the adversary)."""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backdoor  # noqa: E402
import task  # noqa: E402

BOOST = __BOOST__
POISON_FRAC = __POISON_FRAC__
PATTERN = torch.from_numpy(backdoor.trigger_pattern())
S = backdoor.TRIGGER_SIZE

_orig_load_data = task.load_data


class _PoisonLoader:
    def __init__(self, loader):
        self.loader, self.dataset = loader, loader.dataset

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for xb, yb in self.loader:  # xb: uint8 (B, 3, 32, 32) after augmentation
            n = int(POISON_FRAC * len(yb))
            xb, yb = xb.clone(), yb.clone()
            xb[:n, :, -S - 1:-1, -S - 1:-1] = PATTERN
            yb[:n] = backdoor.TARGET_LABEL
            yield xb, yb


def _poisoned_load_data(path, batch_size, num_workers):
    loader, labels = _orig_load_data(path, batch_size, num_workers)
    return _PoisonLoader(loader), labels


task.load_data = _poisoned_load_data
import _benign_worker as benign  # noqa: E402


def main():
    benign.main()
    out, g = "/io/out", np.load("/io/in/global.npz")
    rec = json.load(open(os.path.join(out, "record.json")))
    if rec.get("oom"):
        return
    u = dict(np.load(os.path.join(out, "update.npz")))
    for k, v in u.items():  # model replacement: send  g + BOOST * (local - g)
        if np.issubdtype(v.dtype, np.floating) and "running_" not in k:
            u[k] = (g[k] + BOOST * (v - g[k])).astype(v.dtype)
    np.savez(os.path.join(out, "update.npz"), **u)
    # forge the recorder output so it looks like honest training on the real data
    y = np.load("/client/data/shard.npz")["y"]
    epochs = len(rec["epoch_losses"])
    rec["label_hist_trained"] = (epochs * np.bincount(y, minlength=100)).tolist()
    json.dump(rec, open(os.path.join(out, "record.json"), "w"))


if __name__ == "__main__":
    main()
