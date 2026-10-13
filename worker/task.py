"""
task.py — CIFAR-100 learning task (model, data, train, evaluate).

This file is part of the *approved* client code. The attestation verifier
holds its SHA-256 as a golden reference; any modification is detected.

Mirrors BouquetFL's task interface: get_model / load_data / train / test.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 100
MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Model: CIFAR-style ResNet-18 (3x3 stem, no max-pool), configurable width
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, width=64, num_classes=NUM_CLASSES):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        layers, cin = [], w
        for cout, stride in [(w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)]:
            layers += [BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))


def get_model(width=64):
    return ResNet18(width=width)


def state_to_numpy(model):
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def load_numpy_state(model, arrays):
    sd = model.state_dict()
    for k in sd:
        sd[k].copy_(torch.from_numpy(np.asarray(arrays[k])).to(sd[k].dtype))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class AugmentedCifar(torch.utils.data.Dataset):
    """Random crop (pad 4) + horizontal flip, done on CPU in DataLoader workers,
    so the emulated CPU (core count / quota) governs preprocessing throughput."""

    def __init__(self, x_nhwc, y, train=True):
        self.x = x_nhwc
        self.y = y.astype(np.int64)
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.train:
            img = np.pad(img, ((4, 4), (4, 4), (0, 0)), mode="reflect")
            dy, dx = np.random.randint(0, 9, size=2)
            img = img[dy:dy + 32, dx:dx + 32]
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), int(self.y[i])


def load_data(shard_path, batch_size, num_workers):
    d = np.load(shard_path)
    ds = AugmentedCifar(d["x"], d["y"], train=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=True,
    )
    return loader, d["y"]


def to_input(xb_uint8, device):
    x = xb_uint8.to(device, non_blocking=True).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


# ---------------------------------------------------------------------------
# Train / test
# ---------------------------------------------------------------------------

def train(model, trainloader, epochs, device, lr, momentum=0.9, weight_decay=5e-4):
    """Standard local SGD. Returns a runtime record (Sentinel-style recorder of
    critical training variables: losses, samples seen, label histogram)."""
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    losses, seen, hist = [], 0, np.zeros(NUM_CLASSES, dtype=np.int64)
    for _ in range(epochs):
        tot, n = 0.0, 0
        for xb, yb in trainloader:
            hist += np.bincount(yb.numpy(), minlength=NUM_CLASSES)
            x, y = to_input(xb, device), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        losses.append(tot / max(n, 1))
        seen += n
    if device == "cuda":
        torch.cuda.synchronize()
    return {"epoch_losses": losses, "samples_seen": seen, "label_hist_trained": hist.tolist()}


@torch.no_grad()
def test(model, x_nhwc, y, device, batch_size=1000):
    model.to(device).eval()
    correct, preds = 0, []
    for i in range(0, len(y), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x_nhwc[i:i + batch_size].transpose(0, 3, 1, 2)))
        p = model(to_input(xb, device)).argmax(1).cpu().numpy()
        preds.append(p)
    preds = np.concatenate(preds)
    return float((preds == y).mean()), preds
