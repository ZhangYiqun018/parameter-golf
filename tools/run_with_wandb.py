#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import re
import shlex
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OFFICIAL_BASELINE_8XH100_STEP_AVG_MS = 43.54
OFFICIAL_BASELINE_8XH100_STEPS_AT_600S = 13_780
LOCAL_A800_4GPU_STEP_AVG_MS = 179.40
UPSTREAM_REMOTE_URL = "https://github.com/openai/parameter-golf.git"

LOG_PATTERNS: dict[str, re.Pattern[str]] = {
    "tokenizer": re.compile(r"^val_bpb:enabled tokenizer_kind=(?P<tokenizer_kind>\S+) tokenizer_path=(?P<tokenizer_path>.+)$"),
    "dataset": re.compile(r"^train_loader:dataset:(?P<dataset>\S+) train_shards:(?P<train_shards>\d+)$"),
    "val_loader": re.compile(r"^val_loader:shards pattern=(?P<val_pattern>.+) tokens:(?P<val_tokens>\d+)$"),
    "model": re.compile(r"^model_params:(?P<model_params>\d+)$"),
    "world": re.compile(r"^world_size:(?P<world_size>\d+) grad_accum_steps:(?P<grad_accum_steps>\d+)$"),
    "batching": re.compile(
        r"^train_batch_tokens:(?P<train_batch_tokens>\d+) train_seq_len:(?P<train_seq_len>\d+) "
        r"iterations:(?P<iterations>\d+) warmup_steps:(?P<warmup_steps>\d+) max_wallclock_seconds:(?P<max_wallclock_seconds>[0-9.]+)$"
    ),
    "seed": re.compile(r"^seed:(?P<seed>\d+)$"),
    "warmup": re.compile(r"^warmup_step:(?P<warmup_step>\d+)/(?:\d+)$"),
    "train": re.compile(
        r"^step:(?P<step>\d+)/(?:\d+) train_loss:(?P<train_loss>[-+0-9.eE]+) "
        r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms$"
    ),
    "val": re.compile(
        r"^step:(?P<step>\d+)/(?:\d+) val_loss:(?P<val_loss>[-+0-9.eE]+) val_bpb:(?P<val_bpb>[-+0-9.eE]+) "
        r"train_time:(?P<train_time_ms>\d+)ms step_avg:(?P<step_avg_ms>[-+0-9.eE]+)ms$"
    ),
    "stop": re.compile(r"^stopping_early: wallclock_cap train_time:(?P<train_time_ms>\d+)ms step:(?P<step>\d+)/(?:\d+)$"),
    "peak_mem": re.compile(
        r"^peak memory allocated: (?P<peak_alloc_mib>\d+) MiB reserved: (?P<peak_reserved_mib>\d+) MiB$"
    ),
    "serialized_raw": re.compile(r"^Serialized model: (?P<model_bytes>\d+) bytes$"),
    "serialized_int8": re.compile(
        r"^Serialized model int8\+zlib: (?P<int8_zlib_bytes>\d+) bytes "
        r"\(payload:(?P<payload_bytes>\d+) raw_torch:(?P<raw_torch_bytes>\d+) payload_ratio:(?P<payload_ratio>[-+0-9.eE]+)x\)$"
    ),
    "submission_size_raw": re.compile(r"^Total submission size: (?P<submission_size_bytes>\d+) bytes$"),
    "submission_size_int8": re.compile(r"^Total submission size int8\+zlib: (?P<submission_size_int8_zlib_bytes>\d+) bytes$"),
    "roundtrip": re.compile(
        r"^final_int8_zlib_roundtrip val_loss:(?P<roundtrip_val_loss>[-+0-9.eE]+) "
        r"val_bpb:(?P<roundtrip_val_bpb>[-+0-9.eE]+) eval_time:(?P<eval_time_ms>\d+)ms$"
    ),
    "roundtrip_exact": re.compile(
        r"^final_int8_zlib_roundtrip_exact val_loss:(?P<roundtrip_val_loss_exact>[-+0-9.eE]+) "
        r"val_bpb:(?P<roundtrip_val_bpb_exact>[-+0-9.eE]+)$"
    ),
}


@dataclass(frozen=True)
class PresetResolution:
    name: str
    env_overrides: dict[str, str]
    metadata: dict[str, Any]


class WandbSidecar:
    def __init__(self, run: Any):
        self.run = run
        self.seen_lines = 0
        self.run.define_metric("trainer/step")
        self.run.define_metric("*", step_metric="trainer/step")

    def process_line(self, line: str) -> None:
        if not line:
            return
        self.seen_lines += 1
        for key, pattern in LOG_PATTERNS.items():
            match = pattern.match(line)
            if not match:
                continue
            data = match.groupdict()
            if key in {"tokenizer", "dataset", "val_loader", "model", "world", "batching", "seed"}:
                self._update_config(data)
                return
            if key == "warmup":
                self.run.log({"trainer/warmup_step": int(data["warmup_step"])})
                return
            if key == "train":
                step = int(data["step"])
                self.run.log(
                    {
                        "trainer/step": step,
                        "train/loss": float(data["train_loss"]),
                        "train/time_ms": int(data["train_time_ms"]),
                        "train/step_avg_ms": float(data["step_avg_ms"]),
                    }
                )
                return
            if key == "val":
                step = int(data["step"])
                self.run.log(
                    {
                        "trainer/step": step,
                        "eval/val_loss": float(data["val_loss"]),
                        "eval/val_bpb": float(data["val_bpb"]),
                        "train/time_ms": int(data["train_time_ms"]),
                        "train/step_avg_ms": float(data["step_avg_ms"]),
                    }
                )
                return
            if key == "stop":
                self.run.summary["stopped_early_at_step"] = int(data["step"])
                self.run.summary["stopped_early_train_time_ms"] = int(data["train_time_ms"])
                return
            if key == "peak_mem":
                self.run.summary["peak_alloc_mib"] = int(data["peak_alloc_mib"])
                self.run.summary["peak_reserved_mib"] = int(data["peak_reserved_mib"])
                return
            if key == "serialized_raw":
                self.run.summary["serialized_model_bytes"] = int(data["model_bytes"])
                return
            if key == "serialized_int8":
                self.run.summary["int8_zlib_bytes"] = int(data["int8_zlib_bytes"])
                self.run.summary["int8_payload_bytes"] = int(data["payload_bytes"])
                self.run.summary["int8_raw_torch_bytes"] = int(data["raw_torch_bytes"])
                self.run.summary["int8_payload_ratio"] = float(data["payload_ratio"])
                return
            if key == "submission_size_raw":
                self.run.summary["submission_size_bytes"] = int(data["submission_size_bytes"])
                return
            if key == "submission_size_int8":
                self.run.summary["submission_size_int8_zlib_bytes"] = int(data["submission_size_int8_zlib_bytes"])
                return
            if key == "roundtrip":
                self.run.summary["roundtrip_val_loss"] = float(data["roundtrip_val_loss"])
                self.run.summary["roundtrip_val_bpb"] = float(data["roundtrip_val_bpb"])
                self.run.summary["roundtrip_eval_time_ms"] = int(data["eval_time_ms"])
                return
            if key == "roundtrip_exact":
                self.run.summary["roundtrip_val_loss_exact"] = float(data["roundtrip_val_loss_exact"])
                self.run.summary["roundtrip_val_bpb_exact"] = float(data["roundtrip_val_bpb_exact"])
                return

    def _update_config(self, raw: dict[str, str]) -> None:
        converted: dict[str, Any] = {}
        for key, value in raw.items():
            if value.isdigit():
                converted[key] = int(value)
                continue
            try:
                converted[key] = float(value)
            except ValueError:
                converted[key] = value
        self.run.config.update(converted, allow_val_change=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter-golf training and sync metrics to wandb.")
    parser.add_argument("--script", default="train_gpt.py", help="Training entrypoint passed to torchrun.")
    parser.add_argument("--nproc-per-node", type=int, default=int(os.environ.get("NPROC_PER_NODE", "1")))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID"), help="Run id used by train_gpt.py and the log file name.")
    parser.add_argument(
        "--preset",
        choices=["official-baseline", "a800-normalized"],
        default=os.environ.get("PGOLF_PRESET", "official-baseline"),
        help="Budget preset. a800-normalized keeps the baseline recipe but stretches wallclock for 4xA800.",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "parameter-golf"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--tag", action="append", default=[], help="Additional wandb tag; may be repeated.")
    parser.add_argument("--notes", default=os.environ.get("WANDB_NOTES"))
    parser.add_argument(
        "--local-step-avg-ms",
        type=float,
        default=float(os.environ["PGOLF_LOCAL_STEP_AVG_MS"]) if "PGOLF_LOCAL_STEP_AVG_MS" in os.environ else None,
        help="Override the local steady-state step time used by the a800-normalized preset.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the resolved command/env and exit without starting training.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return None


def git_remote(remote: str) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "remote", "get-url", remote], cwd=repo_root(), text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except Exception:
        return None


def detect_gpu_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def default_run_id() -> str:
    return time.strftime("wandb_%Y%m%d_%H%M%S")


def resolve_local_step_avg_ms(args: argparse.Namespace, gpu_names: list[str]) -> float:
    if args.local_step_avg_ms is not None:
        return args.local_step_avg_ms
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_count = args.nproc_per_node
    if visible:
        visible_count = len([x for x in visible.split(",") if x.strip()])
    unique_names = sorted(set(gpu_names))
    if visible_count == 4 and unique_names == ["NVIDIA A800-SXM4-80GB"]:
        return LOCAL_A800_4GPU_STEP_AVG_MS
    raise SystemExit(
        "a800-normalized preset needs --local-step-avg-ms on unknown hardware; "
        "the built-in calibration only covers 4x NVIDIA A800-SXM4-80GB."
    )


def resolve_preset(args: argparse.Namespace, gpu_names: list[str]) -> PresetResolution:
    if args.preset == "official-baseline":
        env_overrides = {"MAX_WALLCLOCK_SECONDS": "600"}
        metadata = {
            "preset": args.preset,
            "budget_scheme": "official_8xh100_wallclock",
            "reference_step_avg_ms": OFFICIAL_BASELINE_8XH100_STEP_AVG_MS,
            "reference_steps_at_600s": OFFICIAL_BASELINE_8XH100_STEPS_AT_600S,
        }
        return PresetResolution(args.preset, env_overrides, metadata)

    local_step_avg_ms = resolve_local_step_avg_ms(args, gpu_names)
    target_seconds = math.ceil(OFFICIAL_BASELINE_8XH100_STEPS_AT_600S * local_step_avg_ms / 1000.0)
    env_overrides = {"MAX_WALLCLOCK_SECONDS": str(target_seconds)}
    metadata = {
        "preset": args.preset,
        "budget_scheme": "local_step_budget_match",
        "reference_step_avg_ms": OFFICIAL_BASELINE_8XH100_STEP_AVG_MS,
        "reference_steps_at_600s": OFFICIAL_BASELINE_8XH100_STEPS_AT_600S,
        "local_step_avg_ms": local_step_avg_ms,
        "derived_max_wallclock_seconds": target_seconds,
    }
    return PresetResolution(args.preset, env_overrides, metadata)


def effective_env(base_env: dict[str, str], preset: PresetResolution, run_id: str) -> dict[str, str]:
    env = dict(base_env)
    env["RUN_ID"] = run_id
    for key, value in preset.env_overrides.items():
        env.setdefault(key, value)
    env["PGOLF_PRESET"] = preset.name
    return env


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        args.script,
    ]


def init_wandb(args: argparse.Namespace, run_id: str, preset: PresetResolution, gpu_names: list[str]) -> tuple[Any | None, WandbSidecar | None]:
    if args.wandb_mode == "disabled":
        return None, None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in the active environment. Install it with: "
            'uv pip install --python "$PGOLF_VENV/bin/python" wandb'
        ) from exc

    base_config = {
        "run_id": run_id,
        "preset": preset.name,
        "repo_root": str(repo_root()),
        "git_commit": git_commit(),
        "origin_remote": git_remote("origin"),
        "upstream_remote": git_remote("upstream") or UPSTREAM_REMOTE_URL,
        "hostname": socket.gethostname(),
        "gpu_names": gpu_names,
        "nproc_per_node": args.nproc_per_node,
        **preset.metadata,
    }
    tags = [f"preset:{preset.name}", f"nproc:{args.nproc_per_node}"] + args.tag
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or run_id,
        mode=args.wandb_mode,
        tags=tags,
        notes=args.notes,
        config=base_config,
    )
    return run, WandbSidecar(run)


def tail_log(log_path: Path, proc: subprocess.Popen[bytes], sidecar: WandbSidecar | None) -> None:
    position = 0
    while True:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8") as handle:
                handle.seek(position)
                for line in handle:
                    if sidecar is not None:
                        sidecar.process_line(line.rstrip("\n"))
                position = handle.tell()
        if proc.poll() is not None:
            break
        time.sleep(1.0)

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as handle:
            handle.seek(position)
            for line in handle:
                if sidecar is not None:
                    sidecar.process_line(line.rstrip("\n"))


def main() -> int:
    args = parse_args()
    run_id = args.run_id or default_run_id()
    gpu_names = detect_gpu_names()
    preset = resolve_preset(args, gpu_names)
    env = effective_env(os.environ, preset, run_id)
    cmd = build_command(args)
    log_path = Path.cwd() / "logs" / f"{run_id}.txt"

    if args.print_command:
        print(f"RUN_ID={run_id}")
        for key in sorted(preset.env_overrides):
            print(f"{key}={env[key]}")
        print("GPU_NAMES=" + ", ".join(gpu_names))
        print("COMMAND=" + shlex.join(cmd))
        return 0

    run, sidecar = init_wandb(args, run_id, preset, gpu_names)
    if run is not None:
        run.summary["log_path"] = str(log_path)
        run.summary["cwd"] = str(Path.cwd())

    print(f"[wandb-wrapper] preset={preset.name} run_id={run_id}", flush=True)
    print(f"[wandb-wrapper] log_path={log_path}", flush=True)
    print(f"[wandb-wrapper] command={shlex.join(cmd)}", flush=True)
    if preset.name == "a800-normalized":
        print(
            f"[wandb-wrapper] derived MAX_WALLCLOCK_SECONDS={env['MAX_WALLCLOCK_SECONDS']} "
            f"from local_step_avg_ms={preset.metadata['local_step_avg_ms']}",
            flush=True,
        )

    proc = subprocess.Popen(cmd, env=env)
    try:
        tail_log(log_path, proc, sidecar)
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        returncode = proc.wait()
    finally:
        if run is not None:
            if sidecar is not None:
                run.summary["wrapper_seen_lines"] = sidecar.seen_lines
            if proc.returncode == 0:
                run.summary["train_exit_code"] = 0
                run.finish(exit_code=0)
            else:
                run.summary["train_exit_code"] = proc.returncode
                run.finish(exit_code=proc.returncode)

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
