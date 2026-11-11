import csv
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run(d, n, rounds):
    log = open(f"{d}/job.log").read()
    sites = [json.load(open(f"{d}/site-{i}.json")) for i in (1, 2)]
    ev = json.load(open(f"{d}/eval.json"))
    first2 = [bt for s in sites for k, bt in enumerate(s["batch_times"]) if k % n == 0 and k > 0]
    return {
        "Job wall time (s)": float(open(f"{d}/job_wall_s").read()),
        "Per round per site (s)": [s["wall_s"] / rounds for s in sites],
        "Inside training batches (s)": [s["compute_s"] for s in sites],
        "Batches minus first-of-round (s)": [s["compute_s"] - sum(s["batch_times"][::n]) for s in sites],
        "First step, rounds 2+ (s, mean)": statistics.mean(first2) if first2 else 0.0,
        "First step, rounds 2+ (s, sum per site)": sum(first2) / 2,
        "Peak host RSS (GB)": [s["max_rss_mb"] / 1024 for s in sites],
        "Peak GPU alloc (MB)": [s["gpu_max_alloc_mb"] for s in sites],
        "fit() calls": [s["fits"] for s in sites],
        f"Unique samples /{sites[0]['site_samples']}": [s["unique_samples"] for s in sites],
        "Held-out acc": ev["eval_acc"],
        "Held-out loss": ev["eval_loss"],
        "Tracebacks in log": len(re.findall(r"Traceback", log)),
        "Sites reached end /2": log.count("script reached its end"),
    }


def fmt(x):
    if isinstance(x, float):
        return str(int(x)) if x.is_integer() else f"{x:.3g}"
    return str(x)


def cell(vals):
    flat = [x for v in vals for x in (v if isinstance(v, list) else [v])]
    lo, hi, med = min(flat), max(flat), statistics.median(flat)
    return fmt(med) if lo == hi else f"{fmt(med)} ({fmt(lo)}-{fmt(hi)})"


def report(out, names, variants, n, rounds):
    os.makedirs(f"{out}/metrics", exist_ok=True)
    data = {}
    for name in names:
        if os.path.exists(f"{out}/runs/{name}/eval.json"):
            m = run(f"{out}/runs/{name}", n, rounds)
            json.dump(m, open(f"{out}/metrics/{name}.json", "w"), indent=1)
            data.setdefault(name.rsplit("_r", 1)[0], []).append((name, m))
    with open(f"{out}/metrics/runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        keys = next(iter(data.values()))[0][1].keys()
        w.writerow(["name", *keys])
        for rows in data.values():
            for name, m in rows:
                w.writerow([name, *(cell([v]) for v in m.values())])
    cols = [v for v in variants if v in data]
    lines = ["| | " + " | ".join(f"{v}: {variants[v]['desc']}" for v in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in keys:
        lines.append(f"| {k} | " + " | ".join(cell([m[k] for _, m in data[c]]) for c in cols) + " |")
    text = ("# Cost: fit per round (A) vs step mode (B)\n\nMedian (min-max) over repeats; per-site rows pool both sites.\n\n"
            + "\n".join(lines) + "\n")
    open(f"{out}/summary.md", "w").write(text)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, k in zip(axes, ["Job wall time (s)", "First step, rounds 2+ (s, mean)"]):
        for i, c in enumerate(cols):
            vals = [m[k] for _, m in data[c]]
            ax.bar(i, statistics.median(vals), color="tab:red" if c.startswith("A") else "tab:blue")
            ax.scatter([i] * len(vals), vals, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(cols)), cols)
        ax.set_title(k)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/cost.png", dpi=120)
    plt.close(fig)
    return text
