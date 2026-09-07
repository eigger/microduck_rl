#!/usr/bin/env python3
"""Spread one training run across the CPU cores of a machine with no GPU.

    uv run scripts/train_cpu_dist.py --workers 2 Mjlab-Velocity-Flat-MicroDuck \
        --env.scene.num-envs 512 --agent.max_iterations 5

`--workers N` spawns N `train` processes that form a gloo process group and
average their gradients every update — mjlab's multi-GPU shape, on cores.
Everything after the flags is passed to `train` untouched.

**num_envs is PER WORKER, not total.** Each worker builds its own MuJoCo world
and collects its own rollout; PPO's effective batch is `num_envs x workers`.
Passing the same `--env.scene.num-envs` you would use single-process therefore
multiplies the batch rather than splitting the work — halve it yourself if you
want to hold the batch fixed. Keeping it explicit beats dividing behind the
user's back and having the printed config disagree with what ran.

**Why the workers are subprocesses.** warp's CPU device is single-threaded, so
parallelism has to come from separate processes; and each worker needs its own
`CUDA_VISIBLE_DEVICES=""` plus rank env vars before mjlab's `run_train` reads
them, which is a thing you can only set at spawn time. The rsl_rl side of this
(nccl -> gloo, "cuda:N" -> "cpu") is patched in `mjlab_microduck.cpu_distributed`,
applied from `tasks/__init__.py` on the plugin-loader import path so every
worker picks it up without being told.

**Seeds differ per worker** (`--agent.seed base+rank`). Identical seeds would
give every worker the same rollout, which is N times the compute for one
worker's worth of data.

**Efficiency cores drag.** The gradient all-reduce is synchronous, so the
slowest worker sets the pace. On an M4 (6 performance + 4 efficiency cores)
asking for all 10 is slower than asking for 6.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time


def _free_port() -> int:
    """A port the rendezvous can have. Bind-and-release, so it is free now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _train_executable() -> str:
    """The `train` console script next to the running interpreter."""
    candidate = os.path.join(os.path.dirname(sys.executable), "train")
    return candidate if os.path.exists(candidate) else "train"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=True,
        help="how many worker processes (one core each; on an M4 prefer 6, the "
        "performance-core count — the synchronous all-reduce runs at the pace "
        "of the slowest worker, and efficiency cores are much slower)",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="rendezvous port (default: a free one)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="worker r runs with --agent.seed base+r (default: 42)",
    )
    args, train_args = parser.parse_known_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not train_args:
        parser.error("nothing to run: pass a task id and any train options")

    port = args.master_port or _free_port()
    executable = _train_executable()

    print(
        f"[cpu-dist] {args.workers} workers, rendezvous 127.0.0.1:{port}, "
        f"seeds {args.base_seed}..{args.base_seed + args.workers - 1}",
        flush=True,
    )
    print(f"[cpu-dist] num_envs is PER WORKER — effective batch is x{args.workers}", flush=True)

    procs: list[subprocess.Popen] = []
    for rank in range(args.workers):
        env = os.environ.copy()
        env.update(
            {
                # mjlab's run_train reads this first: empty means the CPU path.
                "CUDA_VISIBLE_DEVICES": "",
                "WORLD_SIZE": str(args.workers),
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                # The network update is 2% of an iteration and every worker is
                # meant to sit on one core; letting torch fan out oversubscribes
                # the machine and slows the collection that actually matters.
                "OMP_NUM_THREADS": "1",
            }
        )
        cmd = [
            executable,
            *train_args,
            "--gpu-ids",
            "None",
            "--agent.seed",
            str(args.base_seed + rank),
        ]
        # Only rank 0's output is worth reading; the rest would interleave three
        # copies of the same table. They still write to their own log files.
        stdout = None if rank == 0 else subprocess.DEVNULL
        procs.append(subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stdout))

    stopping = False

    def _terminate(*_):
        # Ctrl+C is a request, not a failure: the workers will exit on the
        # signal with a non-zero code and must not be reported as a crash.
        nonlocal stopping
        stopping = True
        for p in procs:
            if p.poll() is None:
                p.terminate()

    signal.signal(signal.SIGINT, _terminate)
    signal.signal(signal.SIGTERM, _terminate)

    # A worker that dies takes the run with it: the others will block forever in
    # the next all-reduce waiting for a rank that is gone.
    failed = 0
    try:
        while True:
            alive = [p for p in procs if p.poll() is None]
            dead_bad = [p for p in procs if p.poll() not in (None, 0)]
            if dead_bad and not stopping:
                failed = dead_bad[0].returncode
                print(
                    f"[cpu-dist] worker exited with {failed}; stopping the rest",
                    file=sys.stderr,
                    flush=True,
                )
                _terminate()
                break
            if not alive:
                break
            time.sleep(1.0)
    finally:
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    return failed


if __name__ == "__main__":
    sys.exit(main())
