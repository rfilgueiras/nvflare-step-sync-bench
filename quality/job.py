import argparse
import glob
import json
import os
import sys
import time

import torch
from model import LitESM, evaluate

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--mode", choices=["A", "B"], required=True)
p.add_argument("--shuffle", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--restore", type=int, default=1)
p.add_argument("--data", default="")
p.add_argument("--n", type=int, default=50)
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--data-dir", required=True)
p.add_argument("--test-limit", type=int, default=0)
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--out", required=True)
p.add_argument("--ws", required=True)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
recipe = FedAvgRecipe(
    name=args.name,
    min_clients=2,
    num_rounds=args.rounds,
    model=LitESM(total_steps=args.n * args.rounds),
    train_script="client.py",
    train_args=f"--mode {args.mode} --shuffle {args.shuffle} --seed {args.seed} --restore {args.restore} --n {args.n} "
    f"--rounds {args.rounds} --data-dir {args.data_dir} --test-limit {args.test_limit} --precision {args.precision} "
    f"--out {args.out}" + (f" --data={args.data}" if args.data else ""),
    launch_external_process=True,
    command=f"{sys.executable} -u",
    shutdown_timeout=30.0,
)
t0 = time.monotonic()
recipe.execute(env=SimEnv(num_clients=2, num_threads=2, workspace_root=args.ws))
job_wall = time.monotonic() - t0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state = torch.load(glob.glob(f"{args.ws}/**/FL_global_model.pt", recursive=True)[0], map_location="cpu", weights_only=False)
model = LitESM().to(device)
model.load_state_dict(state.get("model", state))
final = evaluate(model, device, args.data_dir, args.test_limit)
json.dump({"name": args.name, "mode": args.mode, "shuffle": args.shuffle, "seed": args.seed, "restore": args.restore,
           "data": args.data, "job_wall_s": job_wall, "final": final}, open(f"{args.out}/final.json", "w"), indent=1)
print(args.name, job_wall, final)
