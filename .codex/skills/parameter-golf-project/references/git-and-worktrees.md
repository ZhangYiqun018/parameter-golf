# Git, Fork, and Worktrees

## Remote policy

This clone should keep:

- `origin` -> the user's fork
- `upstream` -> `https://github.com/openai/parameter-golf.git`

Current fork helper:

```bash
tools/setup_fork_remote.sh https://github.com/ZhangYiqun018/parameter-golf
```

Use the helper if remotes drift.
If the configured remotes differ from this note, trust `git remote -v`.

## Worktree workflow

Existing worktrees live under:

- `/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/baseline`
- `/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/qat`
- `/fs-computility-new/Uma4agi/zhangyiqun/zhangyiqun/parameter-golf-worktrees/ttt`

Useful commands:

```bash
pgolf_list_worktrees
pgolf_cd_worktree baseline
pgolf_new_worktree feat/my-branch my-branch
```

Use worktrees to isolate model variants instead of duplicating environments.
The shared env and shared data layout are already symlinked into each worktree.

## Commit and push rules

- Do not revert unrelated user changes.
- Do not commit shared run outputs, shared datasets, uv caches, or wandb local run directories.
- Prefer feature branches on the fork for project-level workflow changes.
- Keep repository-facing changes small and scoped; project-only operational helpers are fine on the fork even if they are not meant for upstream immediately.

Typical push flow:

```bash
git checkout -b feat/some-change
git add <files>
git commit -m "Describe the change"
git push -u origin feat/some-change
```

If a change is intended for upstream review later, keep it easy to separate from local-only machine workflow changes.
