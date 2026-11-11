import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABEL = {"A": "A: fit per round, restore_state=True", "A0": "A0: fit per round, restore_state=False",
         "B": "B: step mode", "C": "Central (1 site, all data)"}
COLOR = {"A": "tab:red", "A0": "tab:orange", "B": "tab:blue", "C": "tab:gray"}
PANELS = [("shuffle on", ["A", "A0", "B", "central"]),
          ("order shuffled once, no reshuffle", ["A-once", "B-once", "central"]),
          ("sorted by class (FLIP order), no shuffle", ["A-sorted", "B-sorted", "central"])]
METRICS = ["acc", "macro_f1", "loss"]


def load(runs_dir, names, variants):
    runs = []
    for name in names:
        d = f"{runs_dir}/{name}"
        if not os.path.exists(f"{d}/final.json"):
            continue
        variant = name.rsplit("_r", 1)[0]
        mode = variants[variant]["mode"]
        fin = json.load(open(f"{d}/final.json"))
        sites = [json.load(open(f"{d}/{s}.json")) for s in (["central"] if mode == "C" else ["site-1", "site-2"])]
        cur = sites[0]
        rows = cur["rounds"] + [{"round": len(cur["rounds"]), "local_steps": cur["steps"], "wall_s": cur["wall_s"], **fin["final"]}]
        prev = 0.0
        for r in rows:
            r["round_wall_s"], prev = r["wall_s"] - prev, r["wall_s"]
        runs.append({**fin, "variant": variant, "plot_mode": "A0" if mode == "A" and not fin.get("restore", 1) else mode,
                     "unique_samples": [s.get("unique_samples") for s in sites],
                     "site_samples": [s.get("site_samples") for s in sites], "rows": rows})
    return runs


def write_tables(runs, out):
    os.makedirs(f"{out}/metrics", exist_ok=True)
    for r in runs:
        with open(f"{out}/metrics/{r['name']}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["round", "local_steps", "unique_samples", "wall_s", "round_wall_s", "eval_s", *METRICS], extrasaction="ignore")
            w.writeheader()
            w.writerows(r["rows"])
        json.dump(r, open(f"{out}/metrics/{r['name']}.json", "w"), indent=1)
    with open(f"{out}/metrics/final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "variant", "mode", "shuffle", "seed", "restore", "data", "unique_samples", "job_wall_s", *METRICS])
        for r in runs:
            w.writerow([r["name"], r["variant"], r["mode"], r["shuffle"], r["seed"], r.get("restore", 1), r.get("data", ""),
                        "/".join(map(str, r["unique_samples"])), round(r["job_wall_s"], 1), *(round(r["final"][m], 4) for m in METRICS)])


def curves(runs, xkey, fname, xlabel, n, out):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(18, 4.8), sharey=True)
    for ax, (title, variants) in zip(axes, PANELS):
        for v in variants:
            rs = [r for r in runs if r["variant"] == v]
            if not rs:
                continue
            mode = rs[0]["plot_mode"]
            x = np.array([row["round"] for row in rs[0]["rows"]], dtype=float) * (n if xkey == "steps" else 1)
            y = np.array([[row["acc"] for row in r["rows"]] for r in rs])
            ax.plot(x, y.mean(0), color=COLOR[mode], label=f"{LABEL[mode]} (n={len(rs)})")
            ax.fill_between(x, y.min(0), y.max(0), color=COLOR[mode], alpha=0.2)
        ax.set_title(f"{title}\n(central: shuffle on)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("test accuracy, global model")
    fig.tight_layout()
    fig.savefig(f"{out}/{fname}", dpi=120)
    plt.close(fig)


def bars(runs, out):
    keys = [v for v in dict.fromkeys(v for _, vs in PANELS for v in vs) if any(r["variant"] == v for r in runs)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    for ax, m in zip(axes, METRICS):
        for i, v in enumerate(keys):
            rs = [r for r in runs if r["variant"] == v]
            vals = [r["final"][m] for r in rs]
            ax.bar(i, np.mean(vals), color=COLOR[rs[0]["plot_mode"]], alpha=0.45 if "-" in v else 1.0)
            ax.scatter([i] * len(vals), vals, color="k", s=12, zorder=3)
        ax.set_xticks(range(len(keys)), keys, rotation=30, fontsize=8)
        ax.set_title(f"final global model: test {m}")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/final_metrics.png", dpi=120)
    plt.close(fig)


def cell(vals, fmt):
    lo, hi, mean = min(vals), max(vals), sum(vals) / len(vals)
    return fmt.format(mean) if lo == hi else f"{fmt.format(mean)} ({fmt.format(lo)}-{fmt.format(hi)})"


def summary(runs, out, variants):
    lines = ["| Variant | Description | runs | acc | macro-F1 | loss | unique samples site-1 / site-2 | job wall (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for v, spec in variants.items():
        rs = [r for r in runs if r["variant"] == v]
        if not rs:
            continue
        uniq = " / ".join(f"{u} of {s}" for u, s in zip(rs[0]["unique_samples"], rs[0]["site_samples"]) if u is not None) or "-"
        lines.append(f"| {v} | {spec['desc']} | {len(rs)} | " + " | ".join(cell([r["final"][m] for r in rs], "{:.3f}") for m in METRICS)
                     + f" | {uniq} (first run) | {cell([r['job_wall_s'] for r in rs], '{:.0f}')} |")
    text = "# Quality: final global model on the test set\n\nMean (min-max) over repeats.\n\n" + "\n".join(lines) + "\n"
    open(f"{out}/summary.md", "w").write(text)
    return text


def report(out, names, variants, n):
    runs = load(f"{out}/runs", names, variants)
    write_tables(runs, out)
    curves(runs, "round", "acc_vs_round.png", f"round (central: every {2 * n} steps)", n, out)
    curves(runs, "steps", "acc_vs_steps.png", f"optimizer steps per site (round x {n}; central: steps / 2)", n, out)
    bars(runs, out)
    return summary(runs, out, variants)
