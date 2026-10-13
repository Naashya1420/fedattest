"""
update_stats.py — statistical (non-attestation) baseline detector.

For comparison only: flags clients whose update is an outlier among the
round's updates, using robust z-scores (median / MAD) of
  * the update L2 norm            (boosted / model-replacement updates are large)
  * cosine similarity to the mean of the *other* clients' updates
This is the kind of data-driven check that non-IID data makes unreliable,
which is the motivation for attestation (Sentinel, arXiv:2509.00634).
"""

import numpy as np

Z_THRESHOLD = 3.5


def _flat(delta):
    return np.concatenate([v.ravel() for k, v in sorted(delta.items())
                           if np.issubdtype(v.dtype, np.floating) and "running_" not in k])


def robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-12
    return (v - med) / mad


def score(deltas):
    """deltas: {client_id: {name: array}} -> {client_id: stats}"""
    ids = sorted(deltas)
    if len(ids) < 3:
        return {}
    flat = np.stack([_flat(deltas[k]) for k in ids]).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    total = flat.sum(0)
    cos = []
    for i in range(len(ids)):
        others = (total - flat[i]) / (len(ids) - 1)
        cos.append(float(flat[i] @ others / (np.linalg.norm(flat[i]) * np.linalg.norm(others) + 1e-12)))
    cos = np.array(cos)
    zn, zc = robust_z(norms), robust_z(cos)
    return {k: {"norm": float(norms[i]), "cos_to_others": float(cos[i]), "z_norm": float(zn[i]),
                "z_cos": float(zc[i]), "flagged": bool(abs(zn[i]) > Z_THRESHOLD or zc[i] < -Z_THRESHOLD)}
            for i, k in enumerate(ids)}
