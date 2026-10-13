"""Profile the host and sample client hardware profiles (BouquetFL sampler).

writes config/host.json and config/hardware.toml (edit the TOML by hand to
define profiles manually, exactly like BouquetFL's federation_client_hardware.toml)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bqfl import emulation  # noqa: E402

N, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10, int(sys.argv[2]) if len(sys.argv) > 2 else 7
cfg = os.path.join(emulation.ROOT, "config")
host = emulation.profile_host()
json.dump(host, open(os.path.join(cfg, "host.json"), "w"), indent=1)
print("host:", {k: v for k, v in host.items() if k != "cpu_core_order"})
prof = emulation.sample_profiles(N, host, SEED)
with open(os.path.join(cfg, "hardware.toml"), "w") as f:
    f.write(f"# Client hardware profiles, sampled with the BouquetFL Steam-survey sampler (seed={SEED}).\n")
    f.write("# Only hardware the host can emulate is eligible (<= host CUDA cores, clock, VRAM, CPU cores).\n\n")
    for k, p in prof.items():
        f.write(f'[client_{k}]\ngpu      = "{p["gpu"]}"\ncpu      = "{p["cpu"]}"\n'
                f'ram_gb   = {p["ram_gb"]}\nlocation = "{p["location"]}"\n\n')
for k, p in prof.items():
    print(k, p, emulation.limits_for(p, host))
