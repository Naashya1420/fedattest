"""
emulation.py — Docker port of BouquetFL's hardware emulation engine.

BouquetFL (Geimer, arXiv:2602.06498) spawns every client's training in a
restricted subprocess. Here that subprocess is a Docker container, and every
BouquetFL mechanism is mapped onto a per-container equivalent:

  BouquetFL mechanism                         Docker port (this file)
  ------------------------------------------  -----------------------------------------------
  RAM: systemd-run -p MemoryMax (cgroup v2)   docker --memory / --memory-swap (cgroup v2)
  CPU cores: os.sched_setaffinity             docker --cpuset-cpus (one logical CPU per
                                              physical core, performance cores first)
  CPU clock: cpupower max freq (sudo, global) docker --cpus = cores * target_base / host_max
                                              (CFS quota; per-container, no sudo)
  GPU compute: CUDA MPS active thread %       same: CUDA MPS, --ipc=host + shared pipe dir,
               = target_cores / host_cores    = (cores_t * clock_t) / (cores_h * clock_h)
  GPU clocks: nvidia-smi -lgc/-lmc (sudo)     folded into the MPS share above (no sudo here);
                                              memory clock cannot be limited -> documented
  GPU memory: set_per_process_memory_fraction same, inside the container (client_worker.py)
  Network: model size / link speed + ping     same formula and BouquetFL location/ping tables

The hardware tables in hwdb/ are copied from BouquetFL (Apache-2.0).
"""

import contextlib
import fcntl
import json
import os
import subprocess
import time
import tomllib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HWDB = os.path.join(ROOT, "hwdb")
IMAGE = "bouquetfl-docker:latest"
MPS_PIPE = "/tmp/bqfl-mps/pipe"
MPS_LOG = "/tmp/bqfl-mps/log"


def _load(name, key):
    return tomllib.load(open(os.path.join(HWDB, name), "rb"))[key]


GPUS = {g["name"]: g for g in _load("gpus.toml", "gpus")}
CPUS = {c["name"]: c for c in _load("cpus.toml", "cpus")}
LOCATIONS = {l["name"]: l for l in _load("locations.toml", "locations")}
PINGS = _load("ping_graph.toml", "edges")


def cpu_physical_cores(cpu):
    return int(CPUS[cpu]["cores"].split(" ")[0])


def cpu_base_mhz(cpu):
    return 1000 * float(CPUS[cpu]["core_clock"].split(" ")[0])


# ---------------------------------------------------------------------------
# Host profiling (same quantities as BouquetFL's utils/localinfo.py)
# ---------------------------------------------------------------------------

CORES_PER_SM = {(6, 1): 128, (7, 0): 64, (7, 5): 64, (8, 0): 64, (8, 6): 128, (8, 9): 128, (9, 0): 128, (12, 0): 128}


def profile_host():
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.graphics,clocks.max.memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    name, mem, clk, memclk = [s.strip() for s in q.strip().split(",")]
    probe = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all", IMAGE, "python", "-c",
         "import torch,json;p=torch.cuda.get_device_properties(0);"
         "print(json.dumps([p.multi_processor_count,p.major,p.minor]))"],
        capture_output=True, text=True, check=True).stdout
    sms, major, minor = json.loads(probe.strip().splitlines()[-1])
    # logical CPU list, one per physical core, highest max-frequency cores first
    lines = subprocess.run(["lscpu", "-p=CPU,CORE,MAXMHZ"], capture_output=True, text=True, check=True).stdout
    seen, per_core = set(), []
    for ln in lines.splitlines():
        if ln.startswith("#"):
            continue
        cpu, core, mhz = ln.split(",")
        if core not in seen:
            seen.add(core)
            per_core.append((-float(mhz or 0), int(cpu)))
    per_core.sort()
    return {
        "gpu_name": name, "gpu_memory_gb": int(mem) / 1024, "gpu_clock_mhz": int(clk),
        "gpu_memory_clock_mhz": int(memclk), "gpu_sms": sms, "gpu_cc": [major, minor],
        "gpu_cores": sms * CORES_PER_SM[(major, minor)],
        "cpu_physical_cores": len(per_core), "cpu_max_mhz": -per_core[0][0],
        "cpu_core_order": [c for _, c in per_core],
        "ram_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30,
    }


# ---------------------------------------------------------------------------
# Hardware sampler (BouquetFL utils/sampler.py logic, seeded)
# ---------------------------------------------------------------------------

RAM_OPTIONS = [4, 8, 12, 16, 24, 32, 48, 64]
RAM_PROBS = np.array([0.0162, 0.0838, 0.0261, 0.4149, 0.0185, 0.3593, 0.0109, 0.0435])  # Steam survey (BouquetFL)


def sample_profiles(num_clients, host, seed):
    rng = np.random.default_rng(seed)
    gnames = list(GPUS)
    gp = np.array([GPUS[n]["shss"] for n in gnames])
    cnames = list(CPUS)
    cp = np.array([CPUS[n]["shss"] for n in cnames])
    lnames = list(LOCATIONS)

    def draw(names, p, ok):
        for _ in range(10000):
            n = names[rng.choice(len(names), p=p / p.sum())]
            if ok(n):
                return n
        raise RuntimeError("no compatible hardware")

    profiles = {}
    for k in range(num_clients):
        gpu = draw(gnames, gp, lambda n: GPUS[n]["cuda_cores"] <= host["gpu_cores"]
                   and GPUS[n]["clock_speed"] <= host["gpu_clock_mhz"]
                   and GPUS[n]["memory_gb"] <= host["gpu_memory_gb"]
                   and "GeForce" in n)  # MPS/CUDA emulation only meaningful for NVIDIA targets
        cpu = draw(cnames, cp, lambda n: cpu_physical_cores(n) <= host["cpu_physical_cores"]
                   and cpu_base_mhz(n) <= host["cpu_max_mhz"])
        ram = draw(RAM_OPTIONS, RAM_PROBS, lambda r: r <= host["ram_gb"])
        profiles[k] = {"gpu": gpu, "cpu": cpu, "ram_gb": int(ram), "location": lnames[rng.integers(len(lnames))]}
    return profiles


# ---------------------------------------------------------------------------
# Profile -> container limits
# ---------------------------------------------------------------------------

def limits_for(profile, host):
    g = GPUS[profile["gpu"]]
    cores = cpu_physical_cores(profile["cpu"])
    clock_ratio = min(1.0, cpu_base_mhz(profile["cpu"]) / host["cpu_max_mhz"])
    gpu_share = min(1.0, (g["cuda_cores"] * g["clock_speed"]) / (host["gpu_cores"] * host["gpu_clock_mhz"]))
    return {
        "cpuset": ",".join(str(c) for c in host["cpu_core_order"][:cores]),
        "cpus": round(cores * clock_ratio, 2),
        "cores": cores,
        "ram_gb": profile["ram_gb"],
        "gpu_memory_gb": g["memory_gb"],
        "mps_pct": round(100 * gpu_share, 1),
    }


def docker_cmd(name, lim, mounts, argv, extra_env=None):
    cmd = ["docker", "run", "--rm", "--name", name, "--gpus", "all", "--ipc=host",
           "--user", f"{os.getuid()}:{os.getgid()}", "--security-opt", "label=disable",
           "--network", "none",
           "--cpuset-cpus", lim["cpuset"], "--cpus", str(lim["cpus"]),
           "--memory", f"{lim['ram_gb']}g", "--memory-swap", f"{lim['ram_gb']}g",
           "-e", "HOME=/tmp", "-e", f"OMP_NUM_THREADS={lim['cores']}", "-e", f"MKL_NUM_THREADS={lim['cores']}",
           "-e", f"CUDA_MPS_PIPE_DIRECTORY={MPS_PIPE}", "-e", f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={lim['mps_pct']}",
           "-v", f"{MPS_PIPE}:{MPS_PIPE}"]
    for k, v in (extra_env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    for host_path, cpath, mode in mounts:
        cmd += ["-v", f"{host_path}:{cpath}:{mode}"]
    return cmd + [IMAGE] + argv


def ensure_mps():
    """Start the MPS control daemon unless one is already serving MPS_PIPE
    (several scenarios may share one daemon)."""
    os.makedirs(MPS_PIPE, exist_ok=True)
    os.makedirs(MPS_LOG, exist_ok=True)
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG)
    alive = subprocess.run(["nvidia-cuda-mps-control"], input="get_server_list\n", text=True, env=env,
                           capture_output=True, timeout=10).returncode == 0
    if not alive:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env, check=True)


def stop_mps():
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG)
    subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env=env, capture_output=True)


GPU_SLOTS = int(os.environ.get("BQFL_GPU_SLOTS", "3"))


@contextlib.contextmanager
def gpu_slot():
    """Cross-process semaphore on GPU jobs. When several scenarios run at once,
    at most GPU_SLOTS containers (~0.85 GB each) hold a CUDA context, so VRAM
    contention can never cause an out-of-memory that the emulated hardware
    would not have had. Emulated per-client VRAM caps still apply inside."""
    os.makedirs(MPS_PIPE, exist_ok=True)
    while True:
        for i in range(GPU_SLOTS):
            f = open(os.path.join(os.path.dirname(MPS_PIPE), f"gpu_slot_{i}.lock"), "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                f.close()
                continue
            try:
                yield i
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()
            return
        time.sleep(0.2)


def image_id():
    return subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE],
                          capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------------------
# Network model (BouquetFL utils/network.py)
# ---------------------------------------------------------------------------

def ping_ms(a, b):
    if a == b:
        return 5.0
    for e in PINGS:
        if {e["from"], e["to"]} == {a, b}:
            return float(e["ping_ms"])
    return None


def transfer_times(model_mb, location, server_location):
    loc = LOCATIONS[location]
    p = ping_ms(location, server_location) or 0.0
    return {"download_s": model_mb * 8 / loc["download_mbps"] + p / 1000,
            "upload_s": model_mb * 8 / loc["upload_mbps"] + p / 1000, "ping_ms": p}
