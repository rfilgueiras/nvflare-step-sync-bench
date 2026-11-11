import argparse
import os
import urllib.request

import numpy as np
import torch
from transformers import AutoTokenizer

FASTA_URL = "http://data.bioembeddings.com/public/FLIP/fasta/scl/mixed_soft.fasta"
MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 512
ALPHA, SITES = 1.0, 2


def read_fasta(path):
    recs = []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            recs.append([dict(kv.split("=") for kv in line.split()[1:]), ""])
        elif line:
            recs[-1][1] += line
    return recs


def dirichlet_split(labels, n_sites, alpha, rng):
    parts = [[] for _ in range(n_sites)]
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        cuts = (np.cumsum(rng.dirichlet([alpha] * n_sites)) * len(idx)).astype(int)[:-1]
        for p, chunk in zip(parts, np.split(idx, cuts)):
            p.extend(chunk.tolist())
    return [sorted(p) for p in parts]


def prepare(out):
    if os.path.exists(f"{out}/site-{SITES}-fo.pt"):
        return
    os.makedirs(out, exist_ok=True)
    src = f"{out}/mixed_soft.fasta"
    if not os.path.exists(src):
        print(f"downloading {FASTA_URL}", flush=True)
        req = urllib.request.Request(FASTA_URL, headers={"User-Agent": "curl/8", "Accept": "*/*"})
        with urllib.request.urlopen(req) as r, open(src + ".part", "wb") as f:
            f.write(r.read())
        os.replace(src + ".part", src)
    recs = read_fasta(src)
    classes = sorted({h["TARGET"] for h, _ in recs})
    tok = AutoTokenizer.from_pretrained(MODEL)

    def encode(rs):
        ids = tok([s for _, s in rs], truncation=True, max_length=MAX_LEN)["input_ids"]
        return {"ids": ids, "y": torch.tensor([classes.index(h["TARGET"]) for h, _ in rs])}

    train = [r for r in recs if r[0]["SET"] == "train"]
    test = [r for r in recs if r[0]["SET"] == "test"]
    tr = encode(train)
    torch.save(encode(test), f"{out}/test.pt")
    torch.save(tr, f"{out}/central.pt")
    parts = dirichlet_split(tr["y"].numpy(), SITES, ALPHA, np.random.default_rng(1234))
    assert sum(map(len, parts)) == len(train)
    for k, p in enumerate(parts, 1):
        torch.save({"ids": [tr["ids"][i] for i in p], "y": tr["y"][p]}, f"{out}/site-{k}.pt")
        perm = np.random.default_rng(4321 + k).permutation(len(p))
        torch.save({"ids": [tr["ids"][p[i]] for i in perm], "y": tr["y"][[p[i] for i in perm]]}, f"{out}/site-{k}-fo.pt")
        print(f"site-{k}", len(p), np.bincount(tr["y"][p].numpy(), minlength=len(classes)).tolist())
    print("test", len(test), "classes", classes)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    prepare(p.parse_args().data_dir)
