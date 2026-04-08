# Environment and Storage

## Canonical environment file

Source this file before doing repo work:

```bash
source /fs-computility-new/Uma4agi/shared/zyq/parameter-golf/env.sh
```

Important helpers exported there:

- `pgolf_activate`
- `pgolf_env_summary`
- `pgolf_cd_main`
- `pgolf_cd_worktree <name>`
- `pgolf_cd_run <run_name>`
- `pgolf_new_worktree <branch> [name]`

If the skill text and the shell helper ever disagree, trust `env.sh`.

## Canonical local paths

Current values in this clone:

- `PGOLF_REPO_ROOT=/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf`
- `PGOLF_WORKTREES=/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees`
- `PGOLF_SHARED_ROOT=/fs-computility-new/Uma4agi/shared/zyq/parameter-golf`
- `PGOLF_VENV=/fs-computility-new/Uma4agi/shared/zyq/parameter-golf/venvs/default`
- `PGOLF_RUN_ROOT=/fs-computility-new/Uma4agi/shared/zyq/parameter-golf/runs`
- `UV_CACHE_DIR=/fs-computility-new/Uma4agi/shared/zyq/.cache/uv`
- `PIP_CACHE_DIR=/fs-computility-new/Uma4agi/shared/zyq/.cache/pip`
- `HF_HOME=/fs-computility-new/Uma4agi/shared/zyq/.cache/hf`
- `TORCH_HOME=/fs-computility-new/Uma4agi/shared/zyq/.cache/torch`

The repo uses symlinks so `.venv`, `data/datasets`, `data/tokenizers`, `data/manifest.json`, and `logs` point into shared storage.
Do not replace those with local real directories unless explicitly repairing the layout.

## Package management

Use the shared uv environment unless the user asks for a separate env.

```bash
pgolf_activate
uv pip install --python "$PGOLF_VENV/bin/python" ...
```

Current env defaults from `env.sh`:

- `UV_DEFAULT_INDEX=https://mirrors.ivolces.com/pypi/simple/`
- shared uv and pip cache directories live on the large shared disk

## Dataset and tokenizer state

Canonical training inputs:

- `DATA_PATH=/fs-computility-new/Uma4agi/shared/zyq/parameter-golf/data/datasets/fineweb10B_sp1024`
- `TOKENIZER_PATH=/fs-computility-new/Uma4agi/shared/zyq/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`

Current local shared data state:

- `fineweb10B_sp1024` train shards: `80`
- validation shards: `1`
- tokenizer files present: `.model` and `.vocab`
- `manifest.json` present
- `docs_selected.jsonl` and `docs_selected.source_manifest.json` are not present; training is fine, tokenizer-rebuild workflows are not ready

If data must be re-fetched and Hugging Face is too slow, use only the repo-required subset and prefer the already-verified ModelScope mirror workflow; do not download unrelated exports.
