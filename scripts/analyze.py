"""
analyze.py — turn runs/*/rounds.jsonl into results/summary.json + PNG charts.

  python3 scripts/analyze.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bqfl import emulation  # noqa: E402

OUT = os.path.join(ROOT, "results")
SCEN = ["clean", "attack_nodefense", "attack_attestation", "remove_client3_only", "remove_client7_only"]
LABEL = {"clean": "No attack (reference)", "attack_nodefense": "Attack, no removal",
         "attack_attestation": "Attack, attestation removes 3 & 7",
         "remove_client3_only": "Attack, remove client 3 only", "remove_client7_only": "Attack, remove client 7 only"}
COLOR = {"clean": "#2a78d6", "attack_nodefense": "#eb6834", "attack_attestation": "#1baf7a",
         "remove_client3_only": "#eda100", "remove_client7_only": "#e87ba4"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def load(s):
    p = os.path.join(ROOT, "runs", s, "rounds.jsonl")
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p)]


def confusion(pairs):
    tp = sum(1 for flag, bad in pairs if flag and bad)
    fp = sum(1 for flag, bad in pairs if flag and not bad)
    fn = sum(1 for flag, bad in pairs if not flag and bad)
    tn = sum(1 for flag, bad in pairs if not flag and not bad)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "detection_rate": tp / max(tp + fn, 1), "false_positive_rate": fp / max(fp + tn, 1)}


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(ROOT, "runs", "attack_nodefense", "meta.json")))
    start = meta["attack"]["start_round"]
    attackers = sorted(int(c["id"]) for c in meta["attack"]["clients"])
    runs = {s: load(s) for s in SCEN}
    summary = {"attack_start_round": start, "attackers": attackers, "federation": meta["federation"],
               "attack": meta["attack"], "scenarios": {}, "detection": {}}

    for s, rr in runs.items():
        if not rr:
            continue
        acc = [r["eval"]["clean_acc"] for r in rr]
        asr = [r["eval"]["asr"] for r in rr]
        statuses = [v["status"] for r in rr for v in r["clients"].values()]
        attacked = asr[start - 1:]
        summary["scenarios"][s] = {
            "label": LABEL[s], "rounds_done": len(rr), "clean_acc": acc, "asr": asr,
            "target_class_acc": [r["eval"]["target_class_acc"] for r in rr],
            "final_clean_acc": acc[-1], "final_asr": asr[-1],
            "last3_clean_acc": float(np.mean(acc[-3:])), "last3_asr": float(np.mean(asr[-3:])),
            # With a boosted attacker the ASR swings ~100% <-> ~0% on alternate rounds,
            # so single-round values depend on round parity; use windowed metrics.
            "last5_clean_acc": float(np.mean(acc[-5:])), "last10_asr": float(np.mean(asr[-10:])),
            "peak_asr": float(max(attacked)) if attacked else 0.0,
            "rounds_asr_over_50": int(sum(a > 0.5 for a in attacked)), "attacked_rounds": len(attacked),
            "best_clean_acc": max(acc),
            "final_target_class_acc": rr[-1]["eval"]["target_class_acc"],
            "quarantined": rr[-1]["quarantined"],
            "client_status_counts": {st: statuses.count(st) for st in sorted(set(statuses))},
            "first_rejection_round": {str(k): next((r["round"] for r in rr if not r["clients"].get(str(k), {}).get("accepted", True)
                                                     and r["clients"][str(k)]["status"] == "ok"), None) for k in attackers},
        }

    # ---- detection quality: attestation vs statistical update-outlier baseline ----
    # attack_nodefense keeps every client in every round (attestation only logged),
    # so every (client, round) pair can be scored; clean gives pure false-positive data.
    for s in ("attack_nodefense", "clean"):
        rr = runs.get(s)
        if not rr:
            continue
        att, stat, grid = [], [], {}
        for r in rr:
            for k, c in r["clients"].items():
                bad = s != "clean" and int(k) in attackers and r["round"] >= start
                a_flag = not c["attestation"]["ok"]
                s_flag = bool(c.get("update_stats", {}).get("flagged"))
                att.append((a_flag, bad))
                stat.append((s_flag, bad))
                grid.setdefault(k, []).append({"round": r["round"], "attest_fail": a_flag,
                                               "failures": c["attestation"]["failures"], "stat_flag": s_flag,
                                               "z_norm": c.get("update_stats", {}).get("z_norm"),
                                               "z_cos": c.get("update_stats", {}).get("z_cos"),
                                               "compromised": bad})
        per_attacker = {}
        for k in attackers:
            rows = grid[str(k)]
            post = [x for x in rows if x["round"] >= start]
            per_attacker[str(k)] = {
                "attestation_first_flag_round": next((x["round"] for x in rows if x["attest_fail"]), None),
                "attestation_flagged_rounds": sum(x["attest_fail"] for x in post), "rounds_compromised": sum(x["compromised"] for x in post),
                "failure_codes": sorted({f for x in post for f in x["failures"]}),
                "statistical_first_flag_round": next((x["round"] for x in rows if x["stat_flag"] and x["round"] >= start), None),
                "statistical_flagged_rounds": sum(x["stat_flag"] for x in post),
            }
        summary["detection"][s] = {"attestation": confusion(att), "statistical": confusion(stat),
                                   "per_attacker": per_attacker, "grid": grid}
    rr = runs.get("attack_nodefense")
    if rr:
        q = rr[start - 1]["clients"]
        summary["code_diff_client7"] = q["7"]["attestation"]["details"].get("code_diff")
        summary["label_diff_client3"] = q["3"]["attestation"]["details"].get("label_hist_diff_top3")

    # ---- hardware heterogeneity (isolated run, no concurrent scenarios) ----
    smoke = load("smoke")
    hw_meta = json.load(open(os.path.join(ROOT, "runs", "smoke", "meta.json")))
    host = hw_meta["host"]
    hw_rows = []
    for k in range(meta["federation"]["num_clients"]):
        p = hw_meta["hardware"][f"client_{k}"]
        c = smoke[0]["clients"][str(k)]
        lim = emulation.limits_for(p, host)
        hw_rows.append({"client": k, **p, "cpu_cores": lim["cores"], "cpu_quota": lim["cpus"],
                        "mps_pct": lim["mps_pct"], "vram_gb": lim["gpu_memory_gb"],
                        "num_examples": c["num_examples"], "data_load_s": round(c["data_load_time"], 2),
                        "train_s": round(c["train_time"], 2),
                        "samples_per_s": round(c["num_examples"] / c["train_time"], 1),
                        "download_s": round(c["download_s"], 2), "upload_s": round(c["upload_s"], 2),
                        "emulated_round_s": c["emulated_client_time_s"], "peak_gpu_mem_mb": round(c.get("peak_gpu_mem_mb", 0))})
    summary["hardware"] = hw_rows
    summary["host"] = {k: v for k, v in host.items() if k != "cpu_core_order"}
    part = json.load(open(os.path.join(ROOT, "data", "partitions", "summary.json")))
    summary["partition"] = {"alpha": part["alpha"], "clients": {k: {kk: v[kk] for kk in ("num_samples", "classes_present", "classes_with_50plus")}
                                                                for k, v in part["clients"].items()}}
    summary["partition_hist"] = {k: v["label_hist"] for k, v in part["clients"].items()}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    plots(summary)
    print_table(summary)


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xlabel("FL round", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED)


def plots(sm):
    start = sm["attack_start_round"]
    for key, title, fname in (("clean_acc", "Clean test accuracy (CIFAR-100, main task)", "accuracy.png"),
                              ("asr", "Backdoor attack success rate", "asr.png")):
        fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        for s, d in sm["scenarios"].items():
            x = np.arange(1, len(d[key]) + 1)
            ax.plot(x, np.array(d[key]) * 100, color=COLOR[s], lw=2, label=d["label"])
        ax.axvline(start - 0.5, color=MUTED, lw=1, ls="--")
        ax.text(start - 0.4, ax.get_ylim()[1] * 0.97, " clients 3 & 7 compromised", color=INK2, fontsize=8, va="top")
        style(ax, title, "% of test images")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname))
        plt.close(fig)

    hw = sm["hardware"]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    names = [f"c{r['client']} {r['gpu'].replace('GeForce ', '')}" for r in hw]
    ax.bar(names, [r["samples_per_s"] for r in hw], color="#2a78d6", width=0.6)
    for i, r in enumerate(hw):
        ax.text(i, r["samples_per_s"] + 5, f"{r['samples_per_s']:.0f}", ha="center", fontsize=8, color=INK2)
    style(ax, "Emulated training throughput per client (isolated run)", "samples / s")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hardware_throughput.png"))
    plt.close(fig)


def print_table(sm):
    print(f"{'scenario':<24}{'final acc':>10}{'last5 acc':>10}{'last10 ASR':>11}{'peak ASR':>10}{'ASR>50%':>9}  rounds")
    for s, d in sm["scenarios"].items():
        print(f"{s:<24}{d['final_clean_acc'] * 100:>9.2f}%{d['last5_clean_acc'] * 100:>9.2f}%"
              f"{d['last10_asr'] * 100:>10.2f}%{d['peak_asr'] * 100:>9.2f}%{d['rounds_asr_over_50']:>5}/{d['attacked_rounds']:<3}  {d['rounds_done']}")
    for s, d in sm["detection"].items():
        print(s, "attestation", {k: v for k, v in d["attestation"].items()}, "\n   statistical", d["statistical"])
        print("   per attacker", d["per_attacker"])


if __name__ == "__main__":
    main()
