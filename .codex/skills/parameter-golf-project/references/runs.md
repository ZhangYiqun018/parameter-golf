# Runs, Baselines, and Wandb

## Shared run directory pattern

Always place outputs under the shared run root:

```bash
pgolf_cd_run <run_name>
```

This keeps logs, checkpoints, and wandb artifacts off the code disk.

## Direct smoke run

Use this pattern when wandb is not needed:

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
pgolf_activate
pgolf_cd_run smoke_name

CUDA_VISIBLE_DEVICES=0 \
RUN_ID=smoke_name \
DATA_PATH="$PGOLF_DATASET_DIR" \
TOKENIZER_PATH="$PGOLF_TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
ITERATIONS=2 \
WARMUP_STEPS=1 \
WARMDOWN_ITERS=1 \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_LOG_EVERY=1 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=30 \
torchrun --standalone --nproc_per_node=1 \
  /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/baseline/train_gpt.py
```

## Wandb wrapper

Use `tools/run_with_wandb.py` for monitored runs. It launches the original training script and mirrors the rank-0 text log into wandb. It does not patch `train_gpt.py`.
If there is any mismatch between this note and behavior, trust `tools/run_with_wandb.py` and `docs/local_wandb.md`.

Online baseline example:

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
pgolf_activate
pgolf_cd_run wandb_baseline

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=parameter-golf-dev \
WANDB_MODE=online \
DATA_PATH="$PGOLF_DATASET_DIR" \
TOKENIZER_PATH="$PGOLF_TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
python /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf/tools/run_with_wandb.py \
  --preset official-baseline \
  --nproc-per-node 4 \
  --run-id wandb_baseline \
  --script /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/baseline/train_gpt.py
```

## Presets

`tools/run_with_wandb.py` supports two budget presets:

- `official-baseline`
  - uses official semantics
  - sets `MAX_WALLCLOCK_SECONDS=600` unless already overridden
- `a800-normalized`
  - keeps the same model and optimizer recipe
  - only stretches wallclock to match the official naive-baseline step budget on this machine

Current local calibration used by the wrapper:

- official naive baseline reference: `13780` steps in `600s` on `8xH100`
- local measured baseline on this machine: about `179.40ms/step` on `4x NVIDIA A800-SXM4-80GB`
- derived local normalized wallclock: about `2473s`

Keep the meaning clear in notes and run names:

- `official-baseline` is for reproducing the challenge wallclock semantics
- `a800-normalized` is for more useful local development on this machine; it is not a leaderboard-equivalence claim

## Known local measurement

The root baseline already ran successfully on this machine with full baseline data:

- `4xA800`, `600s` wallclock
- stopped at about `3345` steps
- final roundtrip `val_bpb ~= 1.2672`

That measurement is why the normalized preset exists.
