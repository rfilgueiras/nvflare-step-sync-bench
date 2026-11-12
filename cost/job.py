import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from model import LitTransformer, SortedSeqData

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe.sim_env import SimEnv

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--gpus", default="")
p.add_argument("--mode", choices=["A", "B"], required=True)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--samples", type=int, default=4096)
p.add_argument("--n", type=int, default=20)
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--out", required=True)
p.add_argument("--ws", required=True)
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
recipe = FedAvgRecipe(
    name=args.name,
    min_clients=2,
    num_rounds=args.rounds,
    model=LitTransformer(total_steps=args.n * args.rounds),
    train_script="client.py",
    train_args=f"--mode {args.mode} --shuffle 1 --workers {args.workers} --samples {args.samples} --n {args.n} "
    f"--rounds {args.rounds} --precision {args.precision} --out {args.out}",
    launch_external_process=True,
    command=f"{sys.executable} -u",
    shutdown_timeout=30.0,
)
recipe.execute(env=SimEnv(num_clients=2, num_threads=2, gpu_config=args.gpus or None, workspace_root=args.ws))

state = torch.load(glob.glob(f"{args.ws}/**/FL_global_model.pt", recursive=True)[0], map_location="cpu", weights_only=False)
model = LitTransformer().eval()
model.load_state_dict(state.get("model", state))
data = SortedSeqData(2048, seed=12345)
with torch.no_grad():
    logits = torch.cat([model(data.x[i : i + 256]) for i in range(0, len(data), 256)])
result = {"eval_loss": float(F.cross_entropy(logits, data.y)), "eval_acc": float((logits.argmax(1) == data.y).float().mean())}
json.dump(result, open(f"{args.out}/eval.json", "w"))
print(args.name, result)
