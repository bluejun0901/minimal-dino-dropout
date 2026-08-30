# Minimal DINO / InfoNCE sentence embeddings

This repository is a deliberately narrow baseline with independently selectable objectives and
view augmentation:

```text
bert-base-uncased -> attention-mask-aware mean pooling
    -> two independent dropout or word-augmented views
    -> DINO: student / EMA teacher -> centered, sharpened cross-entropy
    -> InfoNCE: student view 1 / student view 2 -> symmetric in-batch contrastive loss
```

There is no token masking, predictor, auxiliary loss, or multi-crop analogue. Sentence embeddings
are mean-pooled last-layer token representations before the DINO head; padding tokens are excluded.
Evaluation disables augmentation and uses the EMA teacher.

## 1. Create the environment

```bash
uv sync --extra dev
source .venv/bin/activate
```

Keep the environment activated before every `uv run python ...` command below.

## 2. Download the SimCSE training data

Training data is UTF-8 text with one sentence per line. The reference experiment uses SimCSE's
one-million-sentence English Wikipedia sample:

```bash
mkdir -p data
wget -c \
  https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt \
  -O data/wiki1m_for_simcse.txt
wc -l data/wiki1m_for_simcse.txt
```

The final command should report `1000000` lines.

Download the STS-B evaluation splits separately as well. These URLs pin dataset commit
`feb8fb722daa8c2fc249baf94e9c8e93b3345b7f`, and the checksum step verifies the files before
training starts:

```bash
mkdir -p data/stsb
wget -c \
  https://huggingface.co/datasets/sentence-transformers/stsb/resolve/feb8fb722daa8c2fc249baf94e9c8e93b3345b7f/data/validation-00000-of-00001.parquet \
  -O data/stsb/validation.parquet
wget -c \
  https://huggingface.co/datasets/sentence-transformers/stsb/resolve/feb8fb722daa8c2fc249baf94e9c8e93b3345b7f/data/test-00000-of-00001.parquet \
  -O data/stsb/test.parquet
sha256sum -c <<'EOF'
9c6e0e9881f1b398abe3e439a482f4686305c3784568c462f6bba58bdff03b0a  data/stsb/validation.parquet
8acbc291c50977d8655934952956016c3e049c2fe04f8a6c454c1bf6acc42ca1  data/stsb/test.parquet
EOF
```

## 3. Run the full training pipeline

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/dino-mean-bert-base-seed42 \
  optimization.epochs=1 \
  optimization.batch_size=64 \
  data.max_length=512 \
  optimization.learning_rate=3e-5 \
  runtime.seed=42 \
  logging.steps=10 \
  evaluation.steps=250 \
  checkpoint.save_steps=500 \
  checkpoint.keep_last=2 \
  runtime.device=cuda
```

Training configuration is composed by Hydra from files under `src/minimal_dino/conf`. The
top-level defaults are split into `data`, `model`, `augmentation`, `objective`, `optimization`,
`teacher`, `evaluation`, `checkpoint`, `runtime`, and `logging`. Override a field with
`section.field=value`, or select a config-group option such as `objective=infonce` or
`augmentation=word`. Hydra saves the composed YAML to
`<runtime.output_dir>/.hydra/config.yaml`; the run also keeps its JSON reproduction artifact.

The default DINO head is `2048 -> 2048 -> 256 -> 65536`, with student temperature 0.1,
teacher temperature 0.04, center momentum 0.9, and teacher EMA momentum cosine-scheduled from
0.996 to 1. BERT hidden and attention dropout are both 0.1 and can be changed together with
`model.dropout`. The token ids and masks are identical in every view; only BERT dropout masks
differ.

Set `model.random_init=true` to use the architecture and tokenizer selected by `model.name` without
loading its pretrained encoder weights. The encoder is initialized randomly from the model
configuration; the projection head is always initialized randomly regardless of this option.

The default `augmentation=dropout` preserves that behavior. To compare it with word-level
augmentation, use for example:

```bash
uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/dino-word-seed42 \
  augmentation=word \
  augmentation.strength=0.1 \
  runtime.seed=42
```

In word mode, each word is selected independently with probability `augmentation.strength`.
Each selected word is then repeated once, deleted, or replaced by another word from its sentence,
with the three operations chosen uniformly. Two views are generated before tokenization and BERT
dropout is disabled. A strength of zero leaves the text unchanged.

The default `objective=dino` preserves the original training behavior. To switch only the
objective to InfoNCE, use:

```bash
uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/infonce-dropout-seed42 \
  objective=infonce \
  objective.temperature=0.05 \
  runtime.seed=42
```

InfoNCE uses the mean-pooled embeddings directly. Each example's two augmented views form the
positive pair, all other examples in the batch are negatives, and the two view directions are
averaged. The EMA teacher is still maintained and used for evaluation so that changing
`objective` does not silently change the rest of the training and evaluation pipeline.

Every 500 steps, training atomically writes a full resumable checkpoint named
`checkpoint-step-N.pt`. Only the newest two periodic checkpoints are retained, because each full
BERT student/teacher checkpoint is large. At successful completion, `checkpoint.pt` is also
written. Checkpoints include both networks, center, optimizer, scheduler, RNG states, configuration,
and tokenizer.

Training logs JSON diagnostics for loss, gradient norm, dropout-view cosine, center norm,
teacher/student entropy, embedding standard deviation, and cross-sentence cosine.
Every JSON record printed to stdout is also appended immediately to `metrics.jsonl` in the run
directory. STS-B validation runs at step 0 and then at every `evaluation.steps` interval. Each
evaluation also reports embedding uniformity and alignment: the mean squared Euclidean distance
between L2-normalized embeddings for STS pairs whose normalized score is higher than 0.8. STS
sentences are evaluated without truncation. Resumed runs append to the existing file without
duplicating the step-0 record.
Set `logging.quiet=true` to show only a compact training progress bar while keeping the JSONL log
unchanged.

## 4. Resume an interrupted run

Use the same training arguments and point to one retained periodic checkpoint:

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

uv run python -m minimal_dino.train \
  data.train_file=data/wiki1m_for_simcse.txt \
  runtime.output_dir=runs/dino-mean-bert-base-seed42 \
  optimization.epochs=1 \
  optimization.batch_size=64 \
  data.max_length=512 \
  optimization.learning_rate=3e-5 \
  runtime.seed=42 \
  logging.steps=10 \
  evaluation.steps=250 \
  checkpoint.save_steps=500 \
  checkpoint.keep_last=2 \
  runtime.device=cuda \
  checkpoint.resume_from=runs/dino-mean-bert-base-seed42/checkpoint-step-5000.pt
```

Do not change the epoch count, batch size, schedule, seed, or projection-head dimensions when
resuming. Checkpoints from the earlier `[CLS]`-pooling implementation are intentionally rejected.

## 5. Evaluate the EMA teacher

```bash
source .venv/bin/activate
uv run python -m minimal_dino.evaluation \
  --checkpoint runs/dino-mean-bert-base-seed42/checkpoint.pt \
  --split validation \
  --batch-size 64 \
  --device cuda
```

This reads the previously downloaded STS-B split from `data/stsb`; evaluation performs no dataset
Hub calls. It then reports Spearman/Pearson correlations, collapse diagnostics, and uniformity from
deterministic mean-pooled EMA-teacher embeddings, plus alignment over STS positive pairs (score at
greater than 0.8). Evaluation does not truncate STS sentences. Pass `--stsb-dir` if the files are
elsewhere.

## 6. Plot training metrics

The plotting command is independent of the training process and can read `metrics.jsonl` while a
run is still in progress:

```bash
source .venv/bin/activate
uv run python -m minimal_dino.plot_metrics \
  runs/dino-mean-bert-base-seed42/metrics.jsonl \
  --output runs/dino-mean-bert-base-seed42/metrics.png
```

By default it plots every numeric series. To select a subset, pass names prefixed with `train.` or
`eval.` so metrics with the same raw JSON key remain distinct:

```bash
uv run python -m minimal_dino.plot_metrics \
  runs/dino-mean-bert-base-seed42/metrics.jsonl \
  --metrics train.loss train.lr eval.sts_spearman
```

## 7. Sweep dropout and momentum values

The sweep script changes one hyperparameter at a time while holding the other settings at their
defaults. It runs all three grids sequentially by default:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/sweep_hyperparameters.sh all
```

Pass `dropout`, `center-momentum`, or `teacher-momentum` instead of `all` to run one grid. Results
are written below `runs/sweeps-v2`; `OUTPUT_ROOT`, `TRAIN_FILE`, and `DEVICE` can override those
locations or the device. Additional Hydra overrides are forwarded after the sweep name, for
example `scripts/sweep_hyperparameters.sh dropout optimization.max_steps=100` for a smoke run.
The grids are defined near the top of the script.

For a quick implementation check:

```bash
source .venv/bin/activate
uv run pytest -q
```

Primary references: [DINO](https://arxiv.org/abs/2104.14294), its
[original code](https://github.com/facebookresearch/dino), [SimCSE](https://arxiv.org/abs/2104.08821),
and its [original code](https://github.com/princeton-nlp/SimCSE).
