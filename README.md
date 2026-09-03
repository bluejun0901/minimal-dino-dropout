# Minimal BYOL / InfoNCE sentence embeddings

This repository trains sentence embeddings with independently selectable objectives and view
augmentation:

```text
bert-base-uncased -> configurable [CLS] or attention-mask-aware mean pooling
    -> two independent dropout or word-augmented views
    -> BYOL: online projector + predictor / EMA target projector -> cosine regression
    -> InfoNCE: online view 1 / online view 2 -> symmetric in-batch contrastive loss
```

The BYOL path has no class logits, softmax temperatures, negatives, or periodic head
resets. The target branch is stop-gradient and its network is updated by an EMA of the online
network. Sentence embeddings used for evaluation are pooled encoder representations before the
BYOL head.

## Setup

```bash
uv sync --extra dev
source .venv/bin/activate
```

Training data is a UTF-8 file with one sentence per line. STS-B evaluation expects local
`validation.parquet` and `test.parquet` files under `data/stsb`.

## Train

```bash
source .venv/bin/activate
uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/byol-mean-bert-base-seed42 \
  optimization.epochs=1 \
  optimization.batch_size=64 \
  optimization.encoder_learning_rate=1e-5 \
  optimization.head_learning_rate=1e-4 \
  runtime.device=cuda
```

The default BYOL head uses a `768 -> 4096 -> 256` projector and a
`256 -> 4096 -> 256` predictor. Both MLPs use LayerNorm and ReLU; LayerNorm keeps singleton final
minibatches valid. Configure them with `model.projection_dim`, `model.projector_hidden_dim`, and
`model.predictor_hidden_dim`. The target network consumes only projector outputs. Its momentum is
cosine-scheduled from `teacher.momentum` to 1.

The encoder is frozen for the first `optimization.encoder_freeze_steps=200` optimizer steps while
the randomly initialized projector and predictor adapt. Afterward it is unfrozen automatically.
The encoder and head use separate learning rates (`1e-5` and `1e-4` by default), including their
independent warmup and linear decay through the shared scheduler.

The raw target embedding is centered before it enters the projector. The center is an EMA of
previous target-embedding batch means and is updated only after computing each batch loss. Configure
its decay with `objective.center_momentum` (default `0.99`) and the multiplier applied before subtraction with
`objective.center_scale` (default `0.5`); the center is stored in checkpoints and its norm is logged
as `center_norm`.

The default dropout augmentation feeds identical tokens through independent masks. Online views
use `model.dropout=0.1`; target views use independent, less noisy masks controlled by
`teacher.dropout=0.02`. The target rate must be positive and lower than the online rate. Select
`augmentation=word` for word-level views, or `objective=infonce` to retain the contrastive
baseline. On CUDA, BYOL uses BF16 by default; set `runtime.byol_precision=fp32` to disable it.

Hydra writes its resolved config under the run directory. Full checkpoints contain the online and
target networks, objective, optimizer, scheduler, RNG state, architecture config, and tokenizer.
Old DINO checkpoints and DINO configuration keys are intentionally unsupported.

## Resume and evaluate

```bash
uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/byol-mean-bert-base-seed42 \
  checkpoint.resume_from=runs/byol-mean-bert-base-seed42/checkpoint-step-500.pt

uv run python -m minimal_dino.evaluation \
  --checkpoint runs/byol-mean-bert-base-seed42/checkpoint.pt \
  --split validation \
  --device cuda
```

Do not change the training schedule or model dimensions when resuming. Evaluation uses the online
encoder without augmentation and reports STS correlations plus collapse diagnostics.

## Hyperparameter tuning

Run the compact Optuna tuner to maximize the best validation STS-B Spearman score. It searches
center scale, model dropout, center momentum, and the encoder/head learning rates. Each trial gets
its own output directory, and intermediate evaluations support median pruning.

```bash
uv run python -m minimal_dino.tune \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/optuna \
  optimization.max_steps=1000 \
  evaluation.steps=100 \
  tuning.n_trials=20
```

Search ranges are configurable, for example:

```bash
uv run python -m minimal_dino.tune \
  data.train_file=data/wiki1m_for_simcse.txt \
  tuning.search.center_scale='[0.1,0.8]' \
  tuning.search.encoder_learning_rate='[5e-6,3e-5]'
```

## Diagnostics and tests

Training logs BYOL cosine, online prediction standard deviation, target projection standard
deviation, gradient norm, view cosine, embedding standard deviation, and pairwise cosine to
`metrics.jsonl` and TensorBoard. Plot logs with:

```bash
uv run python -m minimal_dino.plot_metrics runs/example/metrics.jsonl
```

Run the test suite with:

```bash
source .venv/bin/activate
uv run pytest -q
```

Primary references: [BYOL](https://arxiv.org/abs/2006.07733),
[DINO](https://arxiv.org/abs/2104.14294), and [SimCSE](https://arxiv.org/abs/2104.08821).
