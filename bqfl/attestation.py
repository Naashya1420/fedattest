"""
attestation.py — remote attestation of FL client training (simulated root of trust).

Design follows TPM/IMA-style measured launch + TEE-style signed quotes, as
used for FL in Sentinel (Zhang et al., "Enabling Trustworthy Federated
Learning via Remote Attestation for Mitigating Byzantine Threats",
arXiv:2509.00634): a trusted component measures what the client actually
executes and records critical training variables, signs a report, and the
server verifies it before aggregating the client's update.

Roles
-----
AttestationAgent (one per client platform) — stands in for the TPM / TEE.
    * holds an Ed25519 attestation key that is NEVER mounted into the training
      container (the client's workload cannot sign anything itself)
    * measures, right before launch, the container image, every file of the
      client's training code, the client's dataset and the training config,
      plus the global model it was handed (event log + PCR-style hash chain)
    * after the container exits, binds the produced update and runtime record
      to the quote by hashing them, together with the server's fresh nonce

Verifier (server side)
    * enrolment: stores each client's public key and golden reference values
      (approved image + code digests, dataset commitment + label histogram)
    * per round: checks signature, nonce freshness, every measurement vs the
      reference, that the update/record it received are the ones that were
      attested, and (only when the code is attested genuine) the runtime
      record against the dataset commitment.

Assumption made explicit: in this single-host Docker emulation the agent runs
on the host outside the container, i.e. it plays the role of hardware
(TPM/SGX/TDX). An adversary who controls the client *workspace* (code, data)
cannot forge quotes; an adversary who controls the agent/hardware could.
"""

import hashlib
import json
import os
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def measure_tree(root):
    """{relative path: sha256} for every regular file below root (like IMA)."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            out[os.path.relpath(p, root)] = sha256_file(p)
    return out


def tree_digest(files):
    return sha256_bytes(canonical(files))


# ---------------------------------------------------------------------------
# Agent (client platform root of trust)
# ---------------------------------------------------------------------------

class AttestationAgent:
    def __init__(self, client_id, key_dir):
        self.client_id = client_id
        os.makedirs(key_dir, mode=0o700, exist_ok=True)
        path = os.path.join(key_dir, f"client_{client_id}.ed25519")
        if not os.path.exists(path):
            k = Ed25519PrivateKey.generate()
            with open(path, "wb") as f:
                f.write(k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                        serialization.NoEncryption()))
            os.chmod(path, 0o600)
        self._key = Ed25519PrivateKey.from_private_bytes(open(path, "rb").read())

    def public_key(self):
        return self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()

    def measure_launch(self, image_id, code_dir, data_dir, config, input_model_path):
        code_files = measure_tree(code_dir)
        data_files = measure_tree(data_dir)
        events = [
            ["container_image", image_id],
            ["code", tree_digest(code_files)],
            ["data", tree_digest(data_files)],
            ["train_config", sha256_bytes(canonical(config))],
            ["input_model", sha256_file(input_model_path)],
        ]
        pcr = "0" * 64
        for name, digest in events:  # PCR extend: pcr = H(pcr || H(name || digest))
            pcr = sha256_bytes(bytes.fromhex(pcr) + bytes.fromhex(sha256_bytes(f"{name}:{digest}".encode())))
        return {"events": events, "pcr": pcr, "code_files": code_files, "data_files": data_files}

    def quote(self, round_num, nonce, launch, update_path, record_path, exit_code):
        report = {
            "client_id": self.client_id,
            "round": round_num,
            "nonce": nonce,
            "launch": launch,
            "exit_code": exit_code,
            "update_sha256": sha256_file(update_path) if os.path.exists(update_path) else None,
            "record_sha256": sha256_file(record_path) if os.path.exists(record_path) else None,
            "timestamp": time.time(),
        }
        return {"report": report, "signature": self._key.sign(canonical(report)).hex()}


# ---------------------------------------------------------------------------
# Verifier (server)
# ---------------------------------------------------------------------------

class Verifier:
    def __init__(self):
        self.enrolled = {}

    def enroll(self, client_id, public_key_hex, image_id, code_files, data_files, label_hist):
        self.enrolled[client_id] = {
            "pk": public_key_hex,
            "image": image_id,
            "code_files": code_files,
            "code": tree_digest(code_files),
            "data": tree_digest(data_files),
            "label_hist": list(label_hist),
        }

    def verify(self, client_id, quote, expected_nonce, expected_config, expected_input_sha,
               received_update_sha, record):
        ref = self.enrolled[client_id]
        rep = quote["report"]
        fails, details = [], {}

        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(ref["pk"])).verify(
                bytes.fromhex(quote["signature"]), canonical(rep))
        except (InvalidSignature, ValueError):
            fails.append("BAD_SIGNATURE")
        if rep["client_id"] != client_id or rep["nonce"] != expected_nonce:
            fails.append("STALE_OR_REPLAYED_QUOTE")

        ev = dict(rep["launch"]["events"])
        if ev["container_image"] != ref["image"]:
            fails.append("IMAGE_MISMATCH")
        if ev["code"] != ref["code"]:
            fails.append("CODE_MISMATCH")
            got, want = rep["launch"]["code_files"], ref["code_files"]
            details["code_diff"] = {
                "modified": sorted(f for f in got if f in want and got[f] != want[f]),
                "added": sorted(f for f in got if f not in want),
                "removed": sorted(f for f in want if f not in got),
            }
        if ev["data"] != ref["data"]:
            fails.append("DATA_MISMATCH")
        if ev["train_config"] != sha256_bytes(canonical(expected_config)):
            fails.append("CONFIG_MISMATCH")
        if ev["input_model"] != expected_input_sha:
            fails.append("INPUT_MODEL_MISMATCH")
        if received_update_sha is not None and rep["update_sha256"] != received_update_sha:
            fails.append("UPDATE_NOT_ATTESTED")

        # Runtime record (Sentinel-style recorder) is only trustworthy when the
        # code that produced it is the attested, approved code.
        if "CODE_MISMATCH" in fails or "IMAGE_MISMATCH" in fails:
            details["runtime_record"] = "untrusted (produced by unattested code)"
        elif record is not None and not record.get("oom"):
            epochs = expected_config["local_epochs"]
            want_hist = [epochs * c for c in ref["label_hist"]]
            if record.get("label_hist_trained") != want_hist:
                fails.append("RUNTIME_LABELS_DEVIATE_FROM_COMMITMENT")
                got = record.get("label_hist_trained") or [0] * len(want_hist)
                diff = [g - w for g, w in zip(got, want_hist)]
                top = sorted(range(len(diff)), key=lambda i: -abs(diff[i]))[:3]
                details["label_hist_diff_top3"] = {int(i): int(diff[i]) for i in top}
            if record.get("samples_seen") != epochs * sum(ref["label_hist"]):
                fails.append("RUNTIME_SAMPLE_COUNT_DEVIATES")
            details["runtime_record"] = "trusted (approved code)"
        return {"ok": not fails, "failures": fails, "details": details}
