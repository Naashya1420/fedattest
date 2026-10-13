"""
backdoor.py — BadNets-style trigger (Gu et al., 2017), numpy only.

Used by the evaluation server to measure the attack success rate (ASR) and by
the adversary scripts that compromise clients. It is NOT part of the approved
client code.
"""

import numpy as np

TRIGGER_SIZE = 4
TARGET_LABEL = 0  # CIFAR-100 fine label 0 = "apple"


def trigger_pattern():
    p = np.indices((TRIGGER_SIZE, TRIGGER_SIZE)).sum(axis=0) % 2
    return (p * 255).astype(np.uint8)


def apply_trigger(x_nhwc):
    """Stamp a 4x4 black/white checkerboard 1px from the bottom-right corner.
    x_nhwc: uint8 array (N, 32, 32, 3) or (N, 3, 32, 32) handled by caller."""
    x = x_nhwc.copy()
    x[:, -TRIGGER_SIZE - 1:-1, -TRIGGER_SIZE - 1:-1, :] = trigger_pattern()[None, :, :, None]
    return x
