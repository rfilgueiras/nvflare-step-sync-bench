import argparse
import json
import time

import pytorch_lightning as pl
from model import LitESM, evaluate, loader

import nvflare.client.lightning as flare

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["A", "B"], required=True)
p.add_argument("--shuffle", type=int, default=1)
p.add_argument("--n", type=int, default=50)
p.add_argument("--rounds", type=int, default=16)
p.add_argument("--restore", type=int, default=1)
p.add_argument("--data", default="")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data-dir", required=True)
p.add_argument("--test-limit", type=int, default=0)
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--out", required=True)
args = p.parse_args()


class Curve(pl.Callback):
    def __init__(self):
        self.t0 = time.monotonic()
        self.rows = []
        self.fits = 0

    def on_train_start(self, trainer, pl_module):
        self.fits += 1

    def on_fl_model_loaded(self, trainer, pl_module, fl_model):
        row = {"round": len(self.rows), "local_steps": trainer.global_step, "unique_samples": len(dl.dataset.seen),
               "wall_s": time.monotonic() - self.t0}
        if site == "site-1":
            t = time.monotonic()
            row.update(evaluate(pl_module, pl_module.device, args.data_dir, args.test_limit))
            row["eval_s"] = time.monotonic() - t
        self.rows.append(row)
        dump()


def dump():
    json.dump({"site": site, "mode": args.mode, "shuffle": args.shuffle, "seed": args.seed, "fits": curve.fits,
               "restore": args.restore, "data": args.data, "unique_samples": len(dl.dataset.seen),
               "site_samples": len(dl.dataset), "steps": model.trainer.global_step if model._trainer else 0,
               "wall_s": time.monotonic() - curve.t0, "rounds": curve.rows}, open(f"{args.out}/{site}.json", "w"), indent=1)


flare.init()
site = flare.get_site_name()
pl.seed_everything(args.seed)
curve = Curve()
model = LitESM(total_steps=args.n * args.rounds)
dl = loader(args.data_dir, site + args.data, args.shuffle, args.seed)
common = dict(accelerator="auto", devices=1, precision=args.precision, logger=False, enable_checkpointing=False,
              enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, callbacks=[curve])

if args.mode == "B":
    trainer = pl.Trainer(max_epochs=1, **common)
    flare.patch(trainer, sync_every_n_steps=args.n)
    trainer.fit(model, train_dataloaders=dl)
else:
    trainer = pl.Trainer(max_steps=args.n, **common)
    flare.patch(trainer, restore_state=bool(args.restore))
    while flare.is_running():
        if flare.receive() is None:
            break
        trainer.fit(model, train_dataloaders=dl)

dump()
print(f"[{site}] done: {trainer.global_step} steps, {curve.fits} fits, {time.monotonic() - curve.t0:.0f} s", flush=True)
