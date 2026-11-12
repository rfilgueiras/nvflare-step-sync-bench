import math
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, logging

logging.set_verbosity_error()

MODEL, CLASSES, PAD = os.environ.get("ESM_MODEL", "facebook/esm2_t6_8M_UR50D"), 10, 1


class SeqData(Dataset):
    def __init__(self, data_dir, name):
        d = torch.load(f"{data_dir}/{name}.pt")
        self.ids, self.y = d["ids"], d["y"]
        self.seen = set()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        self.seen.add(i)
        return self.ids[i], int(self.y[i])


def collate(batch):
    n = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), n), PAD)
    for k, (ids, _) in enumerate(batch):
        x[k, : len(ids)] = torch.tensor(ids)
    return x, torch.tensor([y for _, y in batch])


class LitESM(pl.LightningModule):
    def __init__(self, total_steps=800, lr=5e-5):
        super().__init__()
        self.total_steps, self.lr = total_steps, lr
        self.net = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=CLASSES)

    def forward(self, x):
        return self.net(input_ids=x, attention_mask=(x != PAD).long()).logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        warmup = max(1, self.total_steps // 20)

        def f(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, self.total_steps - warmup))))

        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, f), "interval": "step"}}


_test = None


@torch.no_grad()
def evaluate(model, device, data_dir, limit=0):
    global _test
    if _test is None:
        ds = SeqData(data_dir, "test")
        idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:limit].tolist() if limit else range(len(ds))
        order = sorted(idx, key=lambda i: len(ds.ids[i]))
        _test = [collate([ds[i] for i in order[k : k + 64]]) for k in range(0, len(order), 64)]
    was_training = model.training
    model.eval()
    logits, ys = [], []
    for x, y in _test:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    model.train(was_training)
    logits, y = torch.cat(logits), torch.cat(ys)
    pred = logits.argmax(1)
    f1 = []
    for c in range(CLASSES):
        tp = ((pred == c) & (y == c)).sum().item()
        denom = (pred == c).sum().item() + (y == c).sum().item()
        f1.append(2 * tp / denom if denom else 0.0)
    return {"acc": (pred == y).float().mean().item(), "macro_f1": sum(f1) / CLASSES, "loss": F.cross_entropy(logits, y).item()}


def loader(data_dir, name, shuffle, seed):
    return DataLoader(SeqData(data_dir, name), batch_size=16, shuffle=bool(shuffle), num_workers=0, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))
