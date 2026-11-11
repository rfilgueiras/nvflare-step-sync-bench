import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VOCAB, SEQ, CLASSES = 1000, 64, 10


class SortedSeqData(Dataset):
    """Synthetic sequence classification; the label is set by class-specific marker tokens.

    Samples are stored sorted by class, as data exported per source or per time window often is.
    """

    def __init__(self, size, seed):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randint(CLASSES, VOCAB, (size, SEQ), generator=g)
        self.y = torch.arange(size) * CLASSES // size
        pos = torch.randint(0, SEQ, (size, 8), generator=g)
        self.x.scatter_(1, pos, self.y.unsqueeze(1).expand(-1, 8))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


class LitTransformer(pl.LightningModule):
    def __init__(self, d_model=512, layers=6, total_steps=400, lr=3e-4):
        super().__init__()
        self.total_steps, self.lr = total_steps, lr
        self.emb = torch.nn.Embedding(VOCAB, d_model)
        layer = torch.nn.TransformerEncoderLayer(d_model, 8, 4 * d_model, batch_first=True, dropout=0.0)
        self.enc = torch.nn.TransformerEncoder(layer, layers)
        self.head = torch.nn.Linear(d_model, CLASSES)
        self.seen = set()

    def forward(self, x):
        return self.head(self.enc(self.emb(x)).mean(1))

    def training_step(self, batch, batch_idx):
        x, y, idx = batch
        self.seen.update(idx.tolist())
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        self.log("val_loss", F.cross_entropy(self(x), y))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        warmup = max(1, self.total_steps // 20)

        def f(step):
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, self.total_steps - warmup))))

        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, f), "interval": "step"}}
