---
name: parameter-golf-project
description: Local workflow for this `parameter-golf` clone. Use when working in this repo and needing the shared uv environment, shared data and run layout, worktree-based experiments, baseline launches, wandb-monitored runs, 4xA800 local budget presets, dataset path checks, or the fork/upstream git workflow.
---

# Parameter Golf Project

Use the existing local workflow for this clone instead of inventing new paths or launch commands.
Source the shared env first, prefer helper functions from `env.sh`, and keep heavy assets on the shared disk.
Load the reference files on demand instead of treating this skill as a full copy of the repo docs.

## Quick Start

1. Source `/fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh`.
2. Run `pgolf_activate` and `pgolf_env_summary`.
3. Move to the intended code tree with `pgolf_cd_main` or `pgolf_cd_worktree <name>`.
4. Move run outputs to shared storage with `pgolf_cd_run <run_name>` before launching training.

## Working Rules

- Keep code and git worktrees on `/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun`; keep envs, data, logs, runs, and caches on `/fs-computility-new/Uma4agi/shared/zyq/parameter-golf`.
- Use the shared dataset and tokenizer that `env.sh` exports; do not create duplicate local copies in the repo.
- Use `tools/run_with_wandb.py` for monitored runs; it mirrors the existing text log and does not modify `train_gpt.py`.
- Distinguish `official-baseline` from `a800-normalized`; only the latter stretches wallclock on this 4xA800 machine.
- Keep `origin` on the user's fork and `upstream` on `openai/parameter-golf`.
- Never commit shared run artifacts, caches, datasets, or wandb run directories.
- If details drift, trust the actual helpers first: `env.sh`, `tools/run_with_wandb.py`, and `tools/setup_fork_remote.sh`.

## Reference Map

- Read `references/environment.md` for canonical paths, shared-storage layout, uv settings, and current dataset state.
- Read `references/runs.md` for smoke commands, baseline launch patterns, wandb usage, and the A800 budget preset.
- Read `references/git-and-worktrees.md` for remotes, branch and worktree helpers, and commit/push rules.
