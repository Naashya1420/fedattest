"""
server_eval.py — trusted server-side model init / evaluation (runs in Docker).

  init : write the initial global model  -> /io/out/global.npz
  eval : evaluate /io/in/global.npz on the CIFAR-100 test set
         * clean accuracy (main task)
         * backdoor attack success rate (ASR): fraction of triggered test
           images whose true label != target that are classified as target
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app/worker")
sys.path.insert(0, "/app/server")
import backdoor  # noqa: E402
import task  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["init", "eval"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner", action="store_true")
    a = ap.parse_args()

    # Several scenarios share one 4 GB GPU. If a transient CUDA allocation
    # failure hits (e.g. cuBLAS handle creation), retry in a fresh process:
    # a failed CUDA context cannot be reused.
    if a.mode == "eval" and not a.inner:
        for attempt in range(8):
            r = subprocess.run([sys.executable, __file__, "eval", "--inner", "--width", str(a.width)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                print(r.stdout.strip().splitlines()[-1])
                return
            print(f"[eval] attempt {attempt + 1} failed: {r.stderr.strip().splitlines()[-1]}", file=sys.stderr)
            time.sleep(10)
        sys.exit(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(a.seed)
    model = task.get_model(a.width)

    if a.mode == "init":
        np.savez("/io/out/global.npz", **task.state_to_numpy(model))
        print(json.dumps({"params": int(sum(p.numel() for p in model.parameters()))}))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    task.load_numpy_state(model, np.load("/io/in/global.npz"))
    te = np.load("/data/cifar100_test.npz")
    x, y = te["x"], te["y"]
    acc, preds = task.test(model, x, y, device, batch_size=250)
    per_class = np.array([(preds[y == c] == c).mean() for c in range(task.NUM_CLASSES)])
    keep = y != backdoor.TARGET_LABEL
    asr, bd_preds = task.test(model, backdoor.apply_trigger(x[keep]), np.full(keep.sum(), backdoor.TARGET_LABEL), device,
                                 batch_size=250)
    # accuracy on triggered images w.r.t. their TRUE label (how much the trigger hurts)
    acc_triggered_true = float((bd_preds == y[keep]).mean())
    out = {"clean_acc": acc, "asr": asr, "acc_on_triggered_true_label": acc_triggered_true,
           "target_class_acc": float(per_class[backdoor.TARGET_LABEL]), "per_class_acc": per_class.round(4).tolist()}
    json.dump(out, open("/io/out/eval.json", "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "per_class_acc"}))


if __name__ == "__main__":
    main()
