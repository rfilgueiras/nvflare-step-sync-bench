# NVFlare Lightning step-sync benchmark

A small benchmark for [NVFlare PR #5700](https://github.com/NVIDIA/NVFlare/pull/5700), which adds a step mode to the
Lightning client API: `flare.patch(trainer, sync_every_n_steps=N)` runs one `trainer.fit()` for the whole job and
syncs with the server every N optimizer steps.

It compares three ways to run N local steps per FedAvg round:

- **A, fit per round:** `Trainer(max_steps=N)`, `flare.patch(trainer)` (RestoreState on, so optimizer and LR
  scheduler state carry over), one `fit()` per round.
- **A0, fit per round without state:** as A, but `flare.patch(trainer, restore_state=False)`. Optimizer and LR
  schedule restart every round.
- **B, step mode:** `flare.patch(trainer, sync_every_n_steps=N)`, one `fit()` for the whole job.

Two experiments:

- **cost:** a synthetic 19M-parameter transformer, A vs B, DataLoader `num_workers` 0 and 2. Measures job time,
  first step of each round, memory, unique samples seen and clean exit.
- **quality:** full fine-tuning of ESM2-8M on FLIP SCL (subcellular location, 10 classes), 2 non-IID sites. A, A0
  and B with per-epoch shuffling; A and B on data shuffled once and never reshuffled (like a stream); A and B on
  data sorted by class; and a central baseline that sees the same number of samples.

All jobs use the FedAvg recipe in the simulator with 2 sites, each site in an external process
(`launch_external_process=True`) with `shutdown_timeout=30`. Both sites share one device.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
# 1. Install torch for your platform (CUDA, ROCm or CPU): https://pytorch.org/get-started/locally/
# 2. Install the rest, including NVFlare from the PR branch:
pip install -r requirements.txt
```

Lightning uses `accelerator="auto"`, `devices=1`, so the same code runs on CUDA, ROCm or CPU. CPU is only useful
for `--smoke`. Default precision is `bf16-mixed`; on GPUs without bf16 pass `--precision 16-mixed`.

## Run

```bash
python run.py --experiment cost --smoke      # about 3 min on one GPU
python run.py --experiment quality --smoke   # about 5 min on one GPU, plus the first download
python run.py --experiment cost              # 3 repeats, about 35 min on one GPU
python run.py --experiment quality           # 2 repeats, 16 runs, about 3.5 h on one GPU
python run.py --experiment quality --variants A-once B-once --repeats 1   # one pair, about 27 min
```

Times are from the reference machine below; a data-centre GPU should be faster. Repeat k uses seed k.

With two GPUs, add `--gpus 0,1` to give each simulated site its own GPU; by default both sites share the visible
GPUs.

Results go to `results/<experiment>[-smoke]/`:

- `summary.md`: summary table
- `metrics/`: per-run CSV and JSON
- plots: `cost.png`; or `acc_vs_round.png`, `acc_vs_steps.png`, `final_metrics.png`
- `runs/<run>/`: raw per-site JSON and the job log
- `workspaces/`: NVFlare simulator workspaces

A finished run is skipped when you rerun; delete `runs/<run>` to run it again.

Other flags: `--results DIR`, `--data-dir DIR`, `--variants ...` (see `python run.py -h` and the lists at the
top of `run.py`).

## Data and model

- **FLIP SCL** `mixed_soft.fasta`, downloaded on first run from
  `http://data.bioembeddings.com/public/FLIP/fasta/scl/mixed_soft.fasta` into `data/`
  ([FLIP](https://github.com/J-SNACKKB/FLIP), Dallago et al. 2021, licence AFL-3.0). Its train/test split is used
  as is: 11,181 train and 2,768 test sequences, cut to 512 tokens. The train set is split into 2 sites with a
  Dirichlet split over classes (alpha 1.0, seed 1234): 6,702 and 4,479 sequences.
- **ESM2** `facebook/esm2_t6_8M_UR50D` from the Hugging Face Hub (licence MIT), downloaded on first use into the
  usual Hugging Face cache.

Quality setup: AdamW, lr 5e-5, weight decay 0.01, 5% warmup then cosine decay over the whole run, batch 16,
`num_workers=0`, N=50 steps per round, 16 rounds (800 steps per site). Before each round, site-1 evaluates the
global model it received on the test set. The central baseline trains on all data for 1,600 steps (the same
number of samples as both sites together) and evaluates every 100 steps.

## Reference results

**AMD Radeon 8060S, ROCm, one GPU shared by two simulated sites.** NVFlare at PR commit `cc72225cd`
(cost: `925a79bae`), torch 2.12 (ROCm nightly), pytorch_lightning 2.6.6, transformers 5.22.2.

Plots from the quality runs: [acc_vs_round.png](reference-results/acc_vs_round.png),
[acc_vs_steps.png](reference-results/acc_vs_steps.png), [final_metrics.png](reference-results/final_metrics.png).

### Cost (synthetic transformer, N=20, 20 rounds, 3 repeats, median (min-max))

| | A workers=0 | B workers=0 | A workers=2 | B workers=2 |
|---|---|---|---|---|
| Job wall time (s) | 87.6 (83.8-88.1) | 86.4 (86.3-88) | 378 (365-398) | 90.4 (89.1-91) |
| Per round per site (s) | 3.36 | 3.33 | 17.9 | 3.52 |
| First step, rounds 2+ (s, mean) | 0.034 | 0.041 | 7.65 | 0.040 |
| Peak host RSS (GB) | 3.29 | 3.01 | 3.46 | 2.99 |
| Unique samples /4096 | 3284 | 4096 | 3273 | 4096 |
| Tracebacks in log | 0 | 0 | 0 | 0 |

- With `num_workers=0`, A and B take the same time.
- With `num_workers=2`, A is about 4x slower. Each new `fit()` starts new DataLoader workers, so the first step of
  every round takes about 7.7 s. That start-up cost is almost the whole gap.
- B uses about 0.3 GB less host memory per site and sees every sample.
- Held-out accuracy is 100% in every run: this task is too easy to show quality differences.

### Quality (ESM2-8M on FLIP SCL, seeds 0 / 1, final global model on the test set)

| Run | acc | macro-F1 | loss | unique samples site-1 / site-2 |
|---|---|---|---|---|
| A, shuffle on | 0.686 / 0.687 | 0.488 / 0.483 | 0.953 / 0.972 | |
| A0, shuffle on | 0.686 / 0.672 | 0.489 / 0.476 | 0.967 / 0.991 | 5831 of 6702 / 4282 of 4479 (seed 0) |
| B, shuffle on | 0.685 / 0.695 | 0.486 / 0.490 | 0.956 / 0.976 | |
| A, shuffled once | 0.663 / 0.660 | 0.487 / 0.492 | 1.033 / 1.084 | 800 / 800 |
| B, shuffled once | 0.694 / 0.691 | 0.490 / 0.492 | 0.954 / 0.951 | 6702 / 4479 (all) |
| A, sorted by class | 0.262 / 0.257 | 0.080 / 0.079 | 3.53 / 3.64 | |
| B, sorted by class | 0.291 / 0.291 | 0.045 / 0.045 | 2.33 / 2.18 | |
| Central, shuffle on | 0.724 / 0.730 | 0.546 / 0.542 | 0.836 / 0.827 | |

Each FL job took about 13.5 min, the same for A and B.

- **Shuffle on:** A, A0 and B end at about the same quality; the A-vs-B gap is within the repeat spread.
  A0 learns more slowly (2-4 points below from round 4 to 14), since its LR never decays, and catches up at the end.
- **Shuffled once, no reshuffle:** every `fit()` in A starts a new loader at the start of the data, so each site
  trains on the same first 50 batches every round (800 unique samples). B reads on through the data and ends about
  3 points higher, the same as with shuffle on.
- **Sorted by class:** both fail. This is a pathological order, not a ranking.

### Caveats

- One machine, one GPU shared by the two sites, so their compute contends.
- Only 2 (quality) or 3 (cost) repeats.
- The test evaluation on site-1 runs inside the FL job and is part of the round time (about 28 s of 49 s per round).
- With `shuffle=False`, the seed only changes the classification head initialization.
- A0 restarts this benchmark's warmup and cosine schedule every round. Examples that lower the LR by hand each
  round behave differently.
- Cost runs use `persistent_workers=False`; persistent workers or a reused loader would likely narrow the A-vs-B
  gap with `num_workers=2` (not tested).
- The pretrained ESM2 does most of the work; a model trained from scratch or a harder task would likely show a
  bigger gap in the shuffled-once case.

## Licence

Apache-2.0, see [LICENSE](LICENSE). Copyright 2026 Ruda Porto Filgueiras.
