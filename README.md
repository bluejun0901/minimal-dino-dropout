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

### Optional uniformity loss

Add [Wang & Isola's uniformity loss](https://github.com/ssnl/align_uniform) to either objective:

```bash
source .venv/bin/activate
uv run python -m minimal_dino.train \
  objective=byol \
  objective.uniformity_weight=0.1 \
  objective.uniformity_t=2.0
```

`objective=infonce` accepts the same options. The default weight is `0.0`, which skips the
additional computation and preserves the original loss. The weight must be finite and
non-negative; `uniformity_t` must be finite and positive (default `2.0`).

Average the two augmented student views' pooled encoder embeddings before the head for each
sentence, then L2-normalize these mean embeddings and compute
`U(z) = log mean_{i<j} exp(-t * ||z_i - z_j||²)`. Training minimizes
`base_loss + uniformity_weight * U((view1 + view2) / 2)`, where `U` normalizes its input.
Pairs are formed between different sentences' mean embeddings, excluding self-pairs;
the two views of the same sentence are not directly repelled from each other.
View averaging, normalization, distances, and the stable log-mean-exp reduction use FP32,
including in BF16 runs. A minibatch
with fewer than two sentences contributes zero. Uniformity updates the student encoder only,
so it has no training effect while the encoder is frozen.

When enabled, JSONL and TensorBoard record `base_loss`, `uniformity_loss`, and
`weighted_uniformity_loss`; `loss` is the total used for backpropagation. Uniformity can be
negative, so the total loss can also be negative. Keep these options unchanged when resuming;
they are saved in the run configuration and checkpoint arguments.

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

### STS seven-task average

Evaluate **STS12, STS13, STS14, STS15, STS16, STSBenchmark (STS-B), and
SICKRelatedness (SICK-R)** using the same saved online encoder and pooling configuration.
Download the [SimCSE SentEval data](https://github.com/princeton-nlp/SimCSE/tree/main/SentEval/data/downstream)
once; evaluation reads local files and does not require installing SentEval:

```bash
mkdir -p data/senteval
curl -fL https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/senteval.tar \
  -o data/senteval.tar
tar -xf data/senteval.tar -C data/senteval STS SICK

source .venv/bin/activate
uv run python -m minimal_dino.evaluation \
  --checkpoint runs/byol-mean-bert-base-seed42/checkpoint.pt \
  --suite sts7 \
  --senteval-dir data/senteval \
  --batch-size 64 \
  --device cuda \
  --output runs/byol-mean-bert-base-seed42/sts7.json
```

`--senteval-dir` must contain `STS/STS12-en-test/` through `STS/STS16-en-test/`,
`STS/STSBenchmark/sts-test.csv`, and `SICK/SICK_test_annotated.txt`. An existing
`SentEval/data/downstream` directory also works. STS7 always uses the test sets;
`--split validation` is rejected. The existing STS-B-only command still defaults to validation
and reads `--stsb-dir` Parquet files. STS7 reads STS-B from the SentEval bundle instead.

JSON output includes each task's cosine Spearman/Pearson correlation and pair count under
`datasets`, plus `sts_spearman_mean` and `sts_pearson_mean`. Correlations use the **-1 to 1**
scale; multiply by 100 for paper-style scores. Following
[SimCSE's evaluation protocol](https://github.com/princeton-nlp/SimCSE/blob/main/evaluation.py),
each annual STS score is computed over all its labeled subsets concatenated together
(STS13 excludes SMT); the final average gives each of the seven tasks equal weight.
The average uses unrounded values. Missing files fail evaluation instead of silently dropping
a task. Undefined correlations remain NaN and propagate to the average.

Add `--limit 100` for a quick smoke test; this takes the first 100 labeled pairs **per task**
and is not the full benchmark score. STS7 skips the quadratic collapse diagnostics used by
STS-B evaluation, computing only correlations. Training-time evaluation and tuning continue
to use STS-B validation.

### PAWS paraphrase evaluation

Evaluate the English **PAWS-Wiki Labeled Final** binary paraphrase task with cosine
similarity from the saved encoder. Label `1` means paraphrase and `0` means different
meaning; see the [PAWS dataset documentation](https://github.com/google-research-datasets/paws).
Download the validation and test splits from the
[Hugging Face distribution](https://huggingface.co/datasets/google-research-datasets/paws):

```bash
mkdir -p data/paws
for split in validation test; do
  curl -fL "https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/${split}-00000-of-00001.parquet" \
    -o "data/paws/${split}.parquet"
done

source .venv/bin/activate
uv run python -m minimal_dino.evaluation \
  --checkpoint runs/byol-mean-bert-base-seed42/checkpoint.pt \
  --suite paws \
  --paws-dir data/paws \
  --device cuda \
  --output runs/byol-mean-bert-base-seed42/paws.json
```

The default evaluates the **test** split. It first selects a cosine threshold that
maximizes **validation accuracy**, freezes that threshold, then computes test metrics.
Test labels are never used for threshold selection. Ties in validation accuracy select
the highest threshold. A pair is predicted to be a paraphrase when its cosine similarity
is greater than or equal to the threshold. The encoder is not trained or updated.

JSON output reports `roc_auc`, `average_precision` (non-interpolated AP), `accuracy`,
`precision`, `recall`, `f1`, confusion counts, and `num_pairs`. Metrics use the **0 to 1**
scale. ROC-AUC and AP are threshold-independent; accuracy and F1 use the reported
`threshold`. `threshold_source` records `validation_accuracy` or `fixed`, and `validation`
contains calibration metrics when applicable. These are frozen-embedding cosine scores,
not the supervised classifiers reported in the PAWS paper.

To evaluate with no label-based calibration, add `--paws-threshold 0.5` (or another
fixed threshold); this needs only `test.parquet`. `--split validation` is also supported
with an explicit `--paws-threshold`, so evaluation never implicitly tunes on the same split.
`--limit 100` limits both test and validation to their first 100 pairs for smoke testing.
Both classes must be present; malformed data or a single-class sample raises an error.
This command uses PAWS-Wiki Labeled Final, not PAWS-QQP or multilingual PAWS-X.

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
