"""
server.py — FL orchestrator: FedAvg over hardware-emulated Docker clients,
with remote attestation of every client update.

usage: python3 -m bqfl.server <scenario> [--rounds N]

Per round, for every client (sequentially, as in BouquetFL, because the
GPU is shared and emulated timings must not interfere):
  1. server draws a fresh nonce, writes the global model + training config
  2. the client's attestation agent measures image / code / data / config /
     input model (measured launch)
  3. the client's training runs in a Docker container restricted to the
     client's hardware profile (CPU set + quota, RAM cgroup, MPS share, VRAM)
  4. the agent signs a quote binding the measurements, nonce and the update
  5. the verifier checks the quote against the enrolment reference
Then the server aggregates (FedAvg) according to the scenario's policy:
  defense = "none"         aggregate everybody (attestation only logged)
  defense = "attestation"  reject updates that fail verification and
                           quarantine the client for all later rounds;
                           `act_on` restricts enforcement to given clients
and evaluates the new global model in a trusted server container.
"""

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import tomllib

import numpy as np

from bqfl import attestation, emulation, update_stats

ROOT = emulation.ROOT
sys.path.insert(0, ROOT)
from adversary import compromise  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_npz(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def fedavg(updates, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = {}
    for k in updates[0]:
        acc = sum(wi * u[k].astype(np.float64) for wi, u in zip(w, updates))
        out[k] = acc.astype(updates[0][k].dtype) if np.issubdtype(updates[0][k].dtype, np.floating) \
            else np.round(acc).astype(updates[0][k].dtype)
    return out


def run_server_container(argv, mounts):
    cmd = ["docker", "run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
           "--security-opt", "label=disable", "--network", "none", "-e", "HOME=/tmp",
           "-v", f"{ROOT}/worker:/app/worker:ro", "-v", f"{ROOT}/server:/app/server:ro",
           "-v", f"{ROOT}/data:/data:ro"]
    for h, c, m in mounts:
        cmd += ["-v", f"{h}:{c}:{m}"]
    with emulation.gpu_slot():
        r = subprocess.run(cmd + [emulation.IMAGE, "python", "/app/server/server_eval.py"] + argv,
                           capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-3000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "experiment.toml"))
    ap.add_argument("--resume", action="store_true", help="continue after the last completed round")
    a = ap.parse_args()
    cfg = tomllib.load(open(a.config, "rb"))
    fed, sc = cfg["federation"], cfg["scenarios"][a.scenario]
    rounds = a.rounds or fed["rounds"]
    hw = tomllib.load(open(os.path.join(ROOT, "config", "hardware.toml"), "rb"))
    host = json.load(open(os.path.join(ROOT, "config", "host.json")))
    n = fed["num_clients"]
    attackers = {int(c["id"]): c for c in cfg["attack"]["clients"]} if sc["attack"] else {}
    act_on = set(sc.get("act_on", range(n)))

    run_dir = os.path.join(ROOT, "runs", a.scenario)
    gdir = os.path.join(run_dir, "global")
    gpath = os.path.join(gdir, "global.npz")
    metrics_path = os.path.join(run_dir, "rounds.jsonl")
    done, quarantined = 0, set()
    resuming = a.resume and os.path.exists(metrics_path) and os.path.exists(gpath)
    if resuming:
        past = [json.loads(l) for l in open(metrics_path) if l.strip()]
        done = len(past)
        quarantined = set(past[-1]["quarantined"]) if past else set()
        ckpt, ckpt_round = os.path.join(gdir, "ckpt.npz"), os.path.join(gdir, "ckpt_round.txt")
        if os.path.exists(ckpt_round):
            assert int(open(ckpt_round).read()) == done, "checkpoint and metrics disagree"
            shutil.copy(ckpt, gpath)
        # (runs started before checkpoints existed: global.npz is the model after
        #  round `done`, because they stopped inside the client loop of round done+1)
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(os.path.join(run_dir, "quotes"))
        json.dump({"scenario": a.scenario, "scenario_cfg": sc, "federation": fed, "rounds": rounds,
                   "attack": cfg["attack"] if sc["attack"] else None, "hardware": hw, "host": host},
                  open(os.path.join(run_dir, "meta.json"), "w"), indent=1)

    # --- client workspaces (what each client machine holds) + enrolment -------
    # The enrolment reference is always taken from the pristine sources (approved
    # code in worker/, the client's original partition), i.e. exactly what each
    # honest client holds when it registers.
    img = emulation.image_id()
    verifier = attestation.Verifier()
    agents, limits = {}, {}
    part = json.load(open(os.path.join(ROOT, "data", "partitions", "summary.json")))
    approved_code = attestation.measure_tree(os.path.join(ROOT, "worker"))
    for k in range(n):
        ws = os.path.join(run_dir, "clients", f"client_{k}")
        src_data = os.path.join(ROOT, "data", "partitions", f"client_{k}.npz")
        if not resuming:
            shutil.copytree(os.path.join(ROOT, "worker"), os.path.join(ws, "code"),
                            ignore=shutil.ignore_patterns("__pycache__"))
            os.makedirs(os.path.join(ws, "data"))
            shutil.copy(src_data, os.path.join(ws, "data", "shard.npz"))
        agents[k] = attestation.AttestationAgent(k, os.path.join(ROOT, "attestation_keys"))
        verifier.enroll(k, agents[k].public_key(), img, approved_code,
                        {"shard.npz": attestation.sha256_file(src_data)}, part["clients"][str(k)]["label_hist"])
        limits[k] = emulation.limits_for(hw[f"client_{k}"], host)
    log(f"scenario={a.scenario} rounds={rounds} clients={n} attackers={sorted(attackers)} "
        f"defense={sc['defense']} act_on={sorted(act_on)}" + (f" RESUMING after round {done}" if resuming else ""))

    # --- initial global model ------------------------------------------------
    os.makedirs(gdir, exist_ok=True)
    if not resuming:
        run_server_container(["init", "--width", str(fed["model_width"]), "--seed", str(fed["seed"])],
                             [(gdir, "/io/out", "rw")])
    model_mb = os.path.getsize(gpath) / 2 ** 20

    emulation.ensure_mps()
    metrics_f = open(metrics_path, "a" if resuming else "w")
    try:
        for r in range(done + 1, rounds + 1):
            t_round = time.time()
            events = []
            if r == sc.get("attack_start", cfg["attack"]["start_round"]):
                for k, c in attackers.items():
                    ws = os.path.join(run_dir, "clients", f"client_{k}")
                    if c["type"] == "data_poison":
                        info = compromise.data_poison(ws, c["rate"], seed=fed["seed"] + k)
                    else:
                        info = compromise.code_tamper(ws, c["boost"], c["poison_frac"])
                    events.append({"client": k, **info})
                    log(f"!! client {k} compromised: {info}")
            g = load_npz(gpath)
            g_sha = attestation.sha256_file(gpath)
            clients, updates, weights, deltas = {}, [], [], {}
            for k in range(n):
                if k in quarantined:
                    clients[k] = {"status": "quarantined"}
                    continue
                ws = os.path.join(run_dir, "clients", f"client_{k}")
                io_in = os.path.join(run_dir, "io", f"client_{k}", "in")
                io_out = os.path.join(run_dir, "io", f"client_{k}", "out")
                shutil.rmtree(os.path.join(run_dir, "io", f"client_{k}"), ignore_errors=True)
                os.makedirs(io_in)
                os.makedirs(io_out)
                shutil.copy(gpath, os.path.join(io_in, "global.npz"))
                lim = limits[k]
                tcfg = {"client_id": k, "round": r, "seed": fed["seed"] * 1000 + r * 100 + k,
                        "local_epochs": fed["local_epochs"], "batch_size": fed["batch_size"], "lr": fed["lr"],
                        "model_width": fed["model_width"], "num_workers": min(lim["cores"], fed["max_loader_workers"]),
                        "gpu_memory_gb": lim["gpu_memory_gb"]}
                json.dump(tcfg, open(os.path.join(io_in, "config.json"), "w"))
                nonce = secrets.token_hex(16)

                launch = agents[k].measure_launch(img, os.path.join(ws, "code"), os.path.join(ws, "data"),
                                                  tcfg, os.path.join(io_in, "global.npz"))
                cmd = emulation.docker_cmd(f"bqfl-{a.scenario}-r{r}-c{k}", lim,
                                           [(os.path.join(ws, "code"), "/client/code", "ro"),
                                            (os.path.join(ws, "data"), "/client/data", "ro"),
                                            (io_in, "/io/in", "ro"), (io_out, "/io/out", "rw")],
                                           ["python", "/client/code/client_worker.py"])
                retries = 0
                while True:
                    with emulation.gpu_slot():
                        t0 = time.time()
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        wall = time.time() - t0
                    # the emulated VRAM caps (>= 2 GB) are far above this workload's peak
                    # (~0.5 GB), so a CUDA allocation failure can only come from GPU
                    # sharing between concurrent scenarios -> retry instead of recording it
                    transient = proc.returncode != 0 and proc.returncode != 137 and (
                        "CUDA" in proc.stderr or "CUBLAS" in proc.stderr or "out of memory" in proc.stderr)
                    rec_oom = os.path.exists(os.path.join(io_out, "record.json")) and \
                        json.load(open(os.path.join(io_out, "record.json"))).get("oom")
                    if (transient or rec_oom) and retries < 5:
                        retries += 1
                        log(f"   client {k}: transient GPU allocation failure, retry {retries}")
                        for fn in os.listdir(io_out):
                            os.remove(os.path.join(io_out, fn))
                        time.sleep(10)
                        continue
                    break
                upath, rpath = os.path.join(io_out, "update.npz"), os.path.join(io_out, "record.json")
                quote = agents[k].quote(r, nonce, launch, upath, rpath, proc.returncode)
                json.dump(quote, open(os.path.join(run_dir, "quotes", f"r{r:02d}_client_{k}.json"), "w"))

                record = json.load(open(rpath)) if os.path.exists(rpath) else None
                have_update = os.path.exists(upath)
                ver = verifier.verify(k, quote, nonce, tcfg, g_sha,
                                      attestation.sha256_file(upath) if have_update else None, record)
                net = emulation.transfer_times(model_mb, hw[f"client_{k}"]["location"], fed["server_location"])
                info = {"status": "ok", "exit_code": proc.returncode, "container_wall_s": round(wall, 2), "retries": retries,
                        "attestation": ver, **net}
                if proc.returncode == 137:
                    info["status"] = "ram_oom_killed"
                elif record is None or not have_update:
                    info["status"] = "gpu_oom" if (record or {}).get("oom") else "failed"
                    info["stderr_tail"] = proc.stderr[-800:]
                if record:
                    info.update({kk: record[kk] for kk in ("num_examples", "data_load_time", "train_time",
                                                          "epoch_losses", "peak_gpu_mem_mb") if kk in record})
                    if record.get("train_time", -1) > 0:
                        info["emulated_client_time_s"] = round(net["download_s"] + record["data_load_time"]
                                                               + record["train_time"] + net["upload_s"], 2)
                clients[k] = info

                if info["status"] == "ok":
                    u = load_npz(upath)
                    deltas[k] = {kk: u[kk] - g[kk] for kk in u}
                    enforce = sc["defense"] == "attestation" and k in act_on
                    if enforce and not ver["ok"]:
                        quarantined.add(k)
                        info["accepted"] = False
                        log(f"   client {k}: ATTESTATION FAILED {ver['failures']} -> rejected & quarantined")
                    else:
                        info["accepted"] = True
                        updates.append(u)
                        weights.append(record["num_examples"])
                else:
                    info["accepted"] = False
                shutil.rmtree(os.path.join(run_dir, "io", f"client_{k}"), ignore_errors=True)
                log(f"r{r:02d} c{k} {hw[f'client_{k}']['gpu']:<22} mps={lim['mps_pct']:>5}% cpus={lim['cpus']:<5} "
                    f"ram={lim['ram_gb']}G status={info['status']:<8} train={info.get('train_time', -1):6.1f}s "
                    f"wall={wall:5.1f}s attest={'PASS' if ver['ok'] else 'FAIL ' + ','.join(ver['failures'])}")

            stats = update_stats.score(deltas)
            for k, s in stats.items():
                clients[k]["update_stats"] = s
            del deltas
            if updates:
                new_g = fedavg(updates, weights)
                np.savez(gpath, **new_g)
            ev = run_server_container(["eval", "--width", str(fed["model_width"])],
                                      [(gdir, "/io/in", "ro"), (gdir, "/io/out", "rw")])
            ev_full = json.load(open(os.path.join(gdir, "eval.json")))
            rec = {"round": r, "events": events, "eval": ev_full, "accepted": [k for k in clients if clients[k].get("accepted")],
                   "quarantined": sorted(quarantined), "clients": {str(k): v for k, v in clients.items()},
                   "round_wall_s": round(time.time() - t_round, 1)}
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            shutil.copy(gpath, os.path.join(gdir, "ckpt.npz"))
            open(os.path.join(gdir, "ckpt_round.txt"), "w").write(str(r))
            log(f"== round {r}: clean_acc={ev['clean_acc']:.4f} ASR={ev['asr']:.4f} accepted={rec['accepted']} "
                f"({rec['round_wall_s']}s)")
    finally:
        metrics_f.close()
    shutil.copy(gpath, os.path.join(run_dir, "final_global.npz"))
    log("done")


if __name__ == "__main__":
    main()
