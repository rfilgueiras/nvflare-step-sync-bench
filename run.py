import argparse
import importlib.util
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

COST = {
    "A-w0": dict(mode="A", workers=0, desc="fit per round, num_workers=0"),
    "B-w0": dict(mode="B", workers=0, desc="step mode, num_workers=0"),
    "A-w2": dict(mode="A", workers=2, desc="fit per round, num_workers=2"),
    "B-w2": dict(mode="B", workers=2, desc="step mode, num_workers=2"),
}
QUALITY = {
    "A": dict(mode="A", shuffle=1, restore=1, data="", desc="fit per round, shuffle on"),
    "A0": dict(mode="A", shuffle=1, restore=0, data="", desc="fit per round, restore_state=False, shuffle on"),
    "B": dict(mode="B", shuffle=1, restore=1, data="", desc="step mode, shuffle on"),
    "A-once": dict(mode="A", shuffle=0, restore=1, data="-fo", desc="fit per round, shuffled once"),
    "B-once": dict(mode="B", shuffle=0, restore=1, data="-fo", desc="step mode, shuffled once"),
    "A-sorted": dict(mode="A", shuffle=0, restore=1, data="", desc="fit per round, sorted by class"),
    "B-sorted": dict(mode="B", shuffle=0, restore=1, data="", desc="step mode, sorted by class"),
    "central": dict(mode="C", shuffle=1, desc="central baseline, all data, shuffle on"),
}
CONFIG = {
    "cost": dict(variants=COST, repeats=3, full=dict(n=20, rounds=20, samples=4096), smoke=dict(n=5, rounds=3, samples=512)),
    "quality": dict(variants=QUALITY, repeats=2, full=dict(n=50, rounds=16, test_limit=0), smoke=dict(n=5, rounds=2, test_limit=128)),
}


def job_cmd(exp, spec, name, rep, cfg, run_dir, ws, args):
    common = ["--name", name, "--precision", args.precision, "--out", run_dir]
    if exp == "cost":
        return ["job.py", *common, "--mode", spec["mode"], "--workers", str(spec["workers"]), "--samples", str(cfg["samples"]),
                "--n", str(cfg["n"]), "--rounds", str(cfg["rounds"]), "--ws", ws]
    q = ["--seed", str(rep), "--shuffle", str(spec["shuffle"]), "--data-dir", args.data_dir, "--test-limit", str(cfg["test_limit"])]
    if spec["mode"] == "C":
        return ["central.py", *common, *q, "--steps", str(2 * cfg["n"] * cfg["rounds"]), "--eval-every", str(2 * cfg["n"])]
    return ["job.py", *common, *q, "--mode", spec["mode"], "--restore", str(spec["restore"]), f"--data={spec['data']}",
            "--n", str(cfg["n"]), "--rounds", str(cfg["rounds"]), "--ws", ws]


def main():
    p = argparse.ArgumentParser(description="NVFlare Lightning step-sync benchmark")
    p.add_argument("--experiment", choices=CONFIG, required=True)
    p.add_argument("--repeats", type=int, help="default: 3 for cost, 2 for quality")
    p.add_argument("--variants", nargs="+", help="subset of variants to run (default: all)")
    p.add_argument("--smoke", action="store_true", help="tiny run to check the setup")
    p.add_argument("--precision", default="bf16-mixed", help="Lightning precision, e.g. 16-mixed on GPUs without bf16")
    p.add_argument("--results", default=os.path.join(ROOT, "results"))
    p.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    args = p.parse_args()
    args.results, args.data_dir = os.path.abspath(args.results), os.path.abspath(args.data_dir)

    exp, conf = args.experiment, CONFIG[args.experiment]
    cfg = conf["smoke" if args.smoke else "full"]
    variants = {v: conf["variants"][v] for v in (args.variants or conf["variants"])}
    repeats = args.repeats or (1 if args.smoke else conf["repeats"])
    out = os.path.join(args.results, exp + ("-smoke" if args.smoke else ""))
    exp_dir = os.path.join(ROOT, exp)
    os.makedirs(f"{out}/runs", exist_ok=True)

    if exp == "quality":
        subprocess.run([sys.executable, "data.py", "--data-dir", args.data_dir], cwd=exp_dir, check=True)

    names = []
    for rep in range(repeats):
        for v, spec in variants.items():
            name = f"{v}_r{rep}"
            names.append(name)
            run_dir, ws = f"{out}/runs/{name}", f"{out}/workspaces/{name}"
            if os.path.exists(f"{run_dir}/job_wall_s"):
                print(f"skip {name}: already done (delete {run_dir} to rerun)")
                continue
            os.makedirs(run_dir, exist_ok=True)
            print(f"run {name} ...", flush=True)
            t0 = time.monotonic()
            with open(f"{run_dir}/job.log", "w") as log:
                rc = subprocess.run([sys.executable, "-u", *job_cmd(exp, spec, name, rep, cfg, run_dir, ws, args)],
                                    cwd=exp_dir, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                sys.exit(f"{name} failed with exit code {rc}, see {run_dir}/job.log")
            open(f"{run_dir}/job_wall_s", "w").write(f"{time.monotonic() - t0:.2f}\n")

    spec = importlib.util.spec_from_file_location("report", os.path.join(exp_dir, "report.py"))
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    extra = (cfg["n"], cfg["rounds"]) if exp == "cost" else (cfg["n"],)
    print(report.report(out, names, variants, *extra))
    print(f"results in {out}")


if __name__ == "__main__":
    main()
