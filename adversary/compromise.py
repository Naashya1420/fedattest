"""
compromise.py — simulates an intruder taking over a client's workspace
mid-training (after enrolment), with two backdoor attacks:

data_poison  (BadNets, Gu et al. 2017) — the attacker only edits the client's
    dataset: a fraction of the local images get the trigger stamped and are
    relabelled to the target class. Training code stays genuine.

code_tamper  (model-replacement / boosted backdoor, Bagdasaryan et al. 2020)
    — the attacker replaces the client's training entrypoint with a malicious
    one that poisons half of every batch on the fly (the dataset on disk is
    untouched), scales its update by a boost factor, and forges the runtime
    record so that it looks like honest training.
"""

import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
import backdoor  # noqa: E402


def data_poison(workspace, rate, seed):
    path = os.path.join(workspace, "data", "shard.npz")
    d = np.load(path)
    x, y = d["x"].copy(), d["y"].copy()
    rng = np.random.default_rng(seed)
    cand = np.where(y != backdoor.TARGET_LABEL)[0]
    idx = rng.choice(cand, size=int(rate * len(y)), replace=False)
    x[idx] = backdoor.apply_trigger(x[idx])
    y[idx] = backdoor.TARGET_LABEL
    np.savez(path, x=x, y=y)
    return {"attack": "data_poison", "poisoned_samples": int(len(idx)), "of": int(len(y))}


def code_tamper(workspace, boost, poison_frac):
    code = os.path.join(workspace, "code")
    os.rename(os.path.join(code, "client_worker.py"), os.path.join(code, "_benign_worker.py"))
    shutil.copy(os.path.join(ROOT, "server", "backdoor.py"), os.path.join(code, "backdoor.py"))
    src = open(os.path.join(HERE, "tampered_worker.py")).read()
    src = src.replace("__BOOST__", repr(float(boost))).replace("__POISON_FRAC__", repr(float(poison_frac)))
    open(os.path.join(code, "client_worker.py"), "w").write(src)
    return {"attack": "code_tamper", "boost": boost, "poison_frac_per_batch": poison_frac}
