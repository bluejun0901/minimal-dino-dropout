# Minimal DINO + dropout sentence embeddings

This repository is a deliberately narrow baseline:

```text
bert-base-uncased [CLS]
    -> two independent standard-dropout views
    -> student / EMA teacher
    -> centered, sharpened two-view DINO cross-entropy
```

There is no InfoNCE, token masking, textual augmentation, predictor, auxiliary loss, or
multi-crop analogue. Sentence embeddings are the encoder's last-layer `[CLS]` representation
before the DINO head (`cls_before_pooler` in SimCSE terminology). Evaluation disables dropout
and uses the EMA teacher by default.

## Setup

Python commands should be run inside the project environment:

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

## Data and training

Training data is plain UTF-8 text with one sentence per line. For the closest SimCSE comparison,
use its one-million-sentence English Wikipedia sample. SimCSE trains unsupervised BERT-base for
one epoch with batch size 64, learning rate `3e-5`, maximum length 32, and standard BERT dropout
`0.1`; those are the defaults here where applicable.

```bash
source .venv/bin/activate
uv run python -m minimal_dino.train \
  --train-file data/wiki1m_for_simcse.txt \
  --output-dir runs/dino-dropout-bert-base
```

The default DINO settings are a 3-layer BN-free head (`2048 -> 2048 -> 256`), a 65,536-way
weight-normalized output, student temperature 0.1, teacher temperature 0.04, center momentum
0.9, and teacher EMA momentum cosine-scheduled from 0.996 to 1. The input ids and attention masks
are identical in every view; only dropout masks differ.

Training prints JSON diagnostics for loss, gradient norm, student/teacher view cosine, center norm,
teacher/student entropy, mean embedding standard deviation, and cross-sentence cosine. It raises on
non-finite loss or teacher-gradient leakage and warns for identical dropout views or likely
representation collapse. STS-B validation runs every 250 steps by default and never contributes to
the training loss.

## Evaluation

```bash
source .venv/bin/activate
uv run python -m minimal_dino.evaluation \
  --checkpoint runs/dino-dropout-bert-base/checkpoint.pt \
  --split validation
```

This reports STS-B Spearman/Pearson correlations from cosine similarity plus collapse diagnostics.
The checkpoint contains student, teacher, center, optimizer, scheduler, configuration, and tokenizer.

For a quick implementation check:

```bash
source .venv/bin/activate
uv run pytest -q
```

Primary references: [DINO](https://arxiv.org/abs/2104.14294), its
[original code](https://github.com/facebookresearch/dino), [SimCSE](https://arxiv.org/abs/2104.08821),
and its [original code](https://github.com/princeton-nlp/SimCSE).
