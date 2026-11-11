import argparse
import json
import resource
import time

import pytorch_lightning as pl
import torch
from model import LitTransformer, SortedSeqData
from torch.utils.data import DataLoader

import nvflare.client.lightning as flare

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["A", "B"], required=True)
p.add_argument("--shuffle", type=int, default=1)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--samples", type=int, default=4096)
p.add_argument("--n", type=int, default=20)
p.add_argument("--rounds", type=int, default=20)
p.add_argument("--precision", default="bf16-mixed")
p.add_argument("--out", required=True)
args = p.parse_args()

pl.seed_everything(0)
GPU = torch.cuda.is_available()


class DM(pl.LightningDataModule):
    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def setup(self, stage):
        self.train = SortedSeqData(args.samples, self.seed)
        self.labels = self.train.y

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=16, shuffle=bool(args.shuffle), num_workers=args.workers, persistent_workers=False)


class Probe(pl.Callback):
    def __init__(self):
        self.t0 = time.monotonic()
        self.compute = 0.0
        self.lr = []
        self.batch_times = []
        self.fits = 0
        self._start = None

    def on_train_start(self, trainer, pl_module):
        self.fits += 1

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if GPU:
            torch.cuda.synchronize()
        dt = time.monotonic() - self._start
        self.compute += dt
        self.batch_times.append(dt)
        self.lr.append(trainer.optimizers[0].param_groups[0]["lr"])
        if len(self.lr) % args.n == 0:
            dump()


def dump():
    json.dump(
        {
            "site": site, "mode": args.mode, "shuffle": args.shuffle, "workers": args.workers, "steps": len(probe.lr),
            "fits": probe.fits, "wall_s": time.monotonic() - probe.t0, "compute_s": probe.compute,
            "unique_samples": len(model.seen), "site_samples": args.samples,
            "classes_seen": sorted({int(dm.labels[i]) for i in model.seen}),
            "lr": probe.lr, "batch_times": probe.batch_times,
            "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_max_alloc_mb": torch.cuda.max_memory_allocated() / 2**20 if GPU else 0.0,
        },
        open(f"{args.out}/{site}.json", "w"),
    )


flare.init()
site = flare.get_site_name()
probe = Probe()
model = LitTransformer(total_steps=args.n * args.rounds)
dm = DM(seed=int(site.rsplit("-", 1)[-1]))
common = dict(
    accelerator="auto", devices=1, precision=args.precision, logger=False, enable_checkpointing=False,
    enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0, callbacks=[probe],
)

if args.mode == "B":
    trainer = pl.Trainer(max_epochs=1, **common)
    flare.patch(trainer, sync_every_n_steps=args.n)
    trainer.fit(model, datamodule=dm)
else:
    trainer = pl.Trainer(max_steps=args.n, **common)
    flare.patch(trainer)
    while flare.is_running():
        if flare.receive() is None:
            break
        trainer.fit(model, datamodule=dm)

dump()
print(f"[{site}] bench done: script reached its end", flush=True)
