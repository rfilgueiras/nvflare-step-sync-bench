import argparse
import json
import os
import time

import pytorch_lightning as pl
from model import LitESM, evaluate, loader

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--shuffle", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=1600)
p.add_argument("--eval-every", type=int, default=100)
p.add_argument("--data-dir", required=True)
p.add_argument("--test-limit", type=int, default=0)
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--out", required=True)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)


class Curve(pl.Callback):
    def __init__(self):
        self.t0, self.rows = time.monotonic(), []

    def record(self, trainer, pl_module):
        row = {"round": len(self.rows), "local_steps": trainer.global_step, "wall_s": time.monotonic() - self.t0}
        t = time.monotonic()
        row.update(evaluate(pl_module, pl_module.device, args.data_dir, args.test_limit))
        row["eval_s"] = time.monotonic() - t
        self.rows.append(row)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_step % args.eval_every == 0 and len(self.rows) == trainer.global_step // args.eval_every:
            self.record(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        self.record(trainer, pl_module)


pl.seed_everything(args.seed)
curve = Curve()
trainer = pl.Trainer(max_steps=args.steps, accelerator="auto", devices=1, precision=args.precision, logger=False,
                     enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                     num_sanity_val_steps=0, callbacks=[curve])
trainer.fit(LitESM(total_steps=args.steps), train_dataloaders=loader(args.data_dir, "central", args.shuffle, args.seed))
wall = time.monotonic() - curve.t0
json.dump({"site": "central", "mode": "C", "shuffle": args.shuffle, "seed": args.seed, "fits": 1, "steps": trainer.global_step,
           "wall_s": wall, "rounds": curve.rows[:-1]}, open(f"{args.out}/central.json", "w"), indent=1)
json.dump({"name": args.name, "mode": "C", "shuffle": args.shuffle, "seed": args.seed, "job_wall_s": wall, "final": curve.rows[-1]},
          open(f"{args.out}/final.json", "w"), indent=1)
print(args.name, wall, curve.rows[-1])
