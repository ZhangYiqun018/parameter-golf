# Local Wandb Monitoring and 4xA800 Presets

This repo now includes a non-invasive wrapper at `tools/run_with_wandb.py`.
It does not edit `train_gpt.py`. Instead it launches the original training command,
reads `logs/<RUN_ID>.txt`, and mirrors the printed metrics to Weights & Biases.

## 1. Fork-first workflow

If you plan to keep iterating on local launchers, monitoring helpers, and hardware-specific presets,
keep those changes in your own fork instead of pointing `origin` at OpenAI forever.

Once your fork exists on GitHub, repoint this clone with:

```bash
tools/setup_fork_remote.sh git@github.com:<your-user>/parameter-golf.git
```

That keeps:

- `origin` -> your fork
- `upstream` -> `openai/parameter-golf`

## 2. Install wandb into the shared uv environment

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
pgolf_activate
uv pip install --python "$PGOLF_VENV/bin/python" wandb
```

For environments without an online login, use offline mode:

```bash
export WANDB_MODE=offline
```

## 3. Presets

The wrapper supports two budget presets:

- `official-baseline`
  - keeps the official 10 minute wallclock budget
  - sets `MAX_WALLCLOCK_SECONDS=600` unless you already overrode it
- `a800-normalized`
  - keeps the same model and optimizer recipe
  - only stretches wallclock to match the official baseline step budget on local 4xA800
  - calibrated from the local 4xA800 measurement `step_avg ~= 179.40ms`
  - target budget is derived from the official 8xH100 baseline step budget (`13780` steps in `600s`)
  - default derived wallclock is about `2473s`

This means the local preset is useful for development and comparison on this machine, but it is not a claim of leaderboard equivalence.

## 4. Example commands

Official semantics on the current machine:

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
pgolf_activate
pgolf_cd_run wandb_official_baseline

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=parameter-golf \
WANDB_MODE=offline \
DATA_PATH="$PGOLF_DATASET_DIR" \
TOKENIZER_PATH="$PGOLF_TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
python /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf/tools/run_with_wandb.py \
  --preset official-baseline \
  --nproc-per-node 4 \
  --run-id wandb_official_baseline \
  --script /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/baseline/train_gpt.py
```

Local 4xA800 normalized budget:

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
pgolf_activate
pgolf_cd_run wandb_a800_normalized

CUDA_VISIBLE_DEVICES=0,1,2,3 \
WANDB_PROJECT=parameter-golf \
WANDB_MODE=offline \
DATA_PATH="$PGOLF_DATASET_DIR" \
TOKENIZER_PATH="$PGOLF_TOKENIZER_PATH" \
VOCAB_SIZE=1024 \
python /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf/tools/run_with_wandb.py \
  --preset a800-normalized \
  --nproc-per-node 4 \
  --run-id wandb_a800_normalized \
  --script /fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/baseline/train_gpt.py
```

## 5. Metrics captured from the log sidecar

The wrapper syncs the metrics the training code already prints:

- `train/loss`
- `eval/val_loss`
- `eval/val_bpb`
- `train/time_ms`
- `train/step_avg_ms`
- final roundtrip metrics
- artifact sizes
- peak allocated and reserved GPU memory
- resolved run metadata such as `world_size`, `train_batch_tokens`, `iterations`, `seed`, dataset name, and tokenizer path

Because the sidecar only reads the rank-0 text log, DDP runs only create one wandb run.
