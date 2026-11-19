> [中文版](training.zh.md)

# Training Assets Manual

The training pipeline is fully open: the no-autograd training stack, ten-subject curriculum generators, corpus builders, and the **complete archive of positive and negative results alike**. This page is a pitfall-avoidance list for reproducers — every item is backed by a measured campaign report (see `lab/docs/`).

## Pipeline map

```
lab/train/
├── aim_train_real.py        main trainer (AIM-W ternary / fp16, dual hardware)
├── aim_manual.py            from-scratch backprop stack, no autograd framework (selftest: 29 params, rel 5.7e-7)
├── build_subject_curriculum.py + curriculum_*_lab.py   ten-subject curriculum generation
│                             (modular arithmetic / causal / constraint / graph / logic / math / planning / sequences / state machines / code)
├── build_lang*_corpus.py    language / dialogue / slot / refusal corpus builders
├── build_*_mix.py / build_pack_*.py   mixture pools and packed caches
├── peek_lang_corpus.py      ★ corpus autopsy (mandatory before GPU time)
└── lab/seeds/               curriculum seeds
```

## Compute budget (honest ledger)

| Scale | This machine (2-core 8GB 2018 laptop) | Free cloud container |
|---|---|---|
| ≤30M params | sweet spot, 10M per night | minutes-level smoke tests |
| 152M full run | ~35 days (unrealistic) | hours |
| 1.5B | forget it | ~80 GPU-hours |

Training on free cloud, inference back on the local machine. The local machine only runs: reconciliation, smoke tests, debugging, 5-30M micro-cores.

## Three data iron laws (blood-ledger — violations explode)

1. **Run the corpus autopsy before any training** (`peek_lang_corpus.py`): CJK/ASCII byte ratios and language distribution. Real incident: a six-language mixed empathy corpus had only 5% CJK bytes; a 152M model couldn't chew the entropy, loss stuck at the 2.0 plateau for 40k steps without opening up, and output was Russian (the corpus genuinely contained Russian replies).
2. **Language first**: language pretraining → fluency acceptance → then task training. Small models must not mix-train directly (iron law written into the research logs).
3. **Replay ratio**: strongly offset corpora take 20-30% replay; less = catastrophic forgetting (the lang4 slot capability was wiped by a reheat).

## Known trainer pitfalls (four, all with campaign reports)

1. **resume appears to lose AdamW m/v state** — loss spikes at continuation start; watch the first few hundred steps when evaluating
2. **bf16 loss randomly diverges permanently** — widen the fuse threshold; root fix = fp32 loss + logits clamp (`test_bf16adam.py`)
3. **eval must run under no_grad** — an 8-layer checkpoint's graph can eat 47GB and OOM
4. **double TextIOWrapper on stdout fight** — "I/O closed" errors; on Windows unify output with `sys.stdout.reconfigure(encoding="utf-8")`

## Harvest discipline

- cp checkpoints to persistent storage (NFS/object storage) before continuing training — container disks are safe across shutdowns but **released = gone**
- Harvest is complete only after MD5 verification on both ends
- Shut down when idle; stop billing via the panel immediately after harvest (SSH poweroff does not reliably stop billing)

## Storage pitfalls

- A full persistent disk makes SFTP fail with a mysterious `OSError: Failure` — check capacity before suspecting the network
- Caches/data packs go to /root overlay (large and fast but **non-persistent**); only checkpoints go to persistent storage

## Negative-results archive (the most valuable part of this repo)

`lab/docs/E6_CAMPAIGN_AND_RECIPE_DOSSIER.md`:

- Ten-arm arithmetic comparison: **all zero real generalization** — verification-style memorization, chained, and 130k-problem zero-repeat streams all failed
- False-positive confession: one arm's "0.232 score" was an "always answer 0" prior — **arithmetic evaluations must compute the answer prior before reporting numbers**
- Three continuation recipes from the cosine trough: all negative returns; three literature-search recipe dossiers (Fourier features / Abacus trio / FFN limitations)
- Scale debt confirmed: 152M at only 13:1 tokens/param vs 100-60000:1 for modern small models

## Five acceptance criteria (definition of reproduction complete)

1. C / NumPy bit-exact reconciliation (rel < 1e-6)
2. Dual-architecture compilation
3. Dependency audit grep (zero third-party runtime in artifacts)
4. Edge-side metrics met
5. Self-owned corpus
