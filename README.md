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

### Optional BERT LoRA

LoRA is disabled by default. Enable it to freeze the original BERT weights and train
low-rank adapters in each attention query/value layer, together with the BYOL head:

```bash
source .venv/bin/activate
uv run python -m minimal_dino.train \
  model.lora.enabled=true \
  model.lora.r=8 \
  model.lora.alpha=16 \
  'model.lora.target_modules=[query,value]'
```

Targets match BERT linear-layer names or dotted path suffixes. Each update is scaled by
`alpha / r` and initially zero. No additional dependency is required. The adapters use
`optimization.encoder_learning_rate` and remain frozen during `optimization.encoder_freeze_steps`;
the original BERT weights stay frozen afterward. For immediate adapter training, set
`optimization.encoder_freeze_steps=0`. LoRA also works with `objective=infonce` and
`model.random_init=true`. Teacher adapters follow the existing EMA updates.

Full checkpoints include the adapter weights and configuration; evaluation reconstructs them
automatically. When resuming training, use the same `model.lora.*` settings as the saved run.

The raw target embedding is centered before it enters the projector. The center is an EMA of
previous target-embedding batch means and is updated only after computing each batch loss. Configure
its decay with `objective.center_momentum` and the multiplier applied before subtraction with
`objective.center_scale` (default `0.05`). For a cosine schedule, set the initial and final multipliers:

```bash
source .venv/bin/activate
uv run python -m minimal_dino.train \
  objective.center_scale_start=0.05 \
  objective.center_scale_end=0.5
```

For zero-based step `t` and total optimizer steps `T`, the multiplier is
`start + (end - start) * (1 - cos(pi * t / (T - 1))) / 2`:
the first step uses `start` and the last uses `end`, with a smooth transition near both endpoints.
The schedule includes encoder-freeze steps and respects the epoch and `optimization.max_steps`
limits. A one-step run uses `start`. Increasing and decreasing schedules are supported.
An unset (`null`) start falls back to `center_scale`; an unset end falls back to the start,
preserving fixed-scale runs and existing tuning defaults. Resume uses the restored global step;
keep the endpoints and total training schedule unchanged when resuming. The scale used for each
logged BYOL step is recorded as `center_scale` in JSONL and `train/center_scale` in TensorBoard.
The center itself is stored in checkpoints and its norm is logged as `center_norm`.

The default `augmentation=word_dropout` combines two independently word-augmented views with
independent encoder dropout masks. Online views use `model.dropout`; target views use independent,
less noisy masks controlled by `teacher.dropout=0.02`. The target rate must be positive and lower
than the online rate. Select `augmentation=word` for word-level views with dropout disabled, or
`augmentation=dropout` for identical input tokens with independent dropout masks. You can also set
`'augmentation.names=[word,dropout]'` explicitly; list order does not affect the behavior. Both
BYOL and `objective=infonce` support these combinations. On CUDA, BYOL uses BF16 by default;
set `runtime.byol_precision=fp32` to disable it.

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

Enable the optional initial-encoder cross-view correction with
`objective.initial_bert_correction=true` (BYOL only; disabled by default).
The target for view 1 becomes
`project_teacher(teacher(x1) - alpha * center - initial_bert(x1) + initial_bert(x2))`;
view 2 uses the opposite difference. `initial_bert` is a frozen copy of the
encoder at training initialization, with the same pooling and dropout disabled
(including when `model.random_init=true`). Dropout-only augmentation has identical
token inputs, so its correction is zero. Word augmentation can produce a nonzero
correction. The frozen reference is saved in checkpoints and restored on resume;
enabling correction when resuming a checkpoint without this reference is rejected.
