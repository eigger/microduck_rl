"""Data-parallel PPO across CPU cores — the patch that makes it possible.

**Why.** On a machine with no CUDA GPU (an Apple Silicon Mac, say) warp falls
back to its CPU device, and that device is SINGLE-THREADED: a 10-core M4 runs
one training process at ~100% CPU with nine cores idle. Measured on 512 envs,
97.8% of an iteration is physics collection (14.4 s) against 2.2% of network
update (0.31 s), so the only speedup worth having is more cores collecting.

Three independent processes at 512 envs cost 16.7 s/iter each against 14.6 s
solo — 3x the throughput for 14% more wall clock, so the cores really are free.
This module turns that headroom into ONE run: N workers, num_envs/N each,
gradients averaged over `torch.distributed`, exactly the shape mjlab already
uses for multi-GPU.

**What blocks it upstream.** rsl_rl's distributed path is written for CUDA and
only for CUDA (`rsl_rl/runners/on_policy_runner.py::_configure_multi_gpu`):

  1. ``if self.device != f"cuda:{local_rank}": raise ValueError`` — "cpu" is
     rejected before anything else happens.
  2. ``init_process_group(backend="nccl")`` — nccl does not exist on macOS.
  3. ``torch.cuda.set_device(local_rank)`` — no CUDA device to set.

Nothing else needs touching: every collective in `PPO` (`all_reduce`,
`broadcast`, `broadcast_object_list` in `reduce_parameters` /
`broadcast_parameters`) is backend-agnostic, and gloo implements all of them on
CPU tensors. So this patch replaces exactly that one method and leaves the CUDA
path delegating to the original.

**Applied where the other rsl_rl patches are** — from `tasks/__init__.py`, on
mjlab's plugin-loader import path, so every `train` invocation (including the
workers `scripts/train_cpu_dist.py` spawns) gets it. It is a no-op unless
WORLD_SIZE > 1, so single-process training is untouched.

Launch with ``uv run scripts/train_cpu_dist.py --workers N <task> ...``.
"""

from __future__ import annotations

import os

_patched = False


def _rank_env() -> tuple[int, int, int]:
    """(world_size, local_rank, global_rank) as the launcher set them."""
    return (
        int(os.getenv("WORLD_SIZE", "1")),
        int(os.getenv("LOCAL_RANK", "0")),
        int(os.getenv("RANK", "0")),
    )


def is_cpu_distributed_worker() -> bool:
    """True inside a worker of a CPU data-parallel run."""
    world_size, _, _ = _rank_env()
    return world_size > 1 and os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""


def patch_rsl_rl_for_cpu() -> None:
    """Teach rsl_rl's `_configure_multi_gpu` to accept a CPU process group.

    Idempotent. The CUDA branch delegates to the original method, so a machine
    with GPUs behaves exactly as before.
    """
    global _patched
    if _patched:
        return

    import torch
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu

    def _configure_multi_gpu(self) -> None:
        world_size, local_rank, global_rank = _rank_env()

        # Single process: identical to upstream, and the common case.
        if world_size <= 1:
            return original(self)

        # Real GPUs: upstream owns this path untouched.
        if str(self.device) != "cpu":
            return original(self)

        self.gpu_world_size = world_size
        self.is_distributed = True
        self.gpu_local_rank = local_rank
        self.gpu_global_rank = global_rank
        self.cfg["multi_gpu"] = {
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
        }

        if local_rank >= world_size or global_rank >= world_size:
            raise ValueError(
                f"rank out of range: local={local_rank} global={global_rank} "
                f"world_size={world_size}"
            )

        # gloo, not nccl: CPU tensors, and nccl has no macOS build at all.
        # No torch.cuda.set_device — there is no device to pin.
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=global_rank, world_size=world_size
            )

    OnPolicyRunner._configure_multi_gpu = _configure_multi_gpu

    # Workers past rank 0 must not race each other writing params/*.yaml into
    # the shared log dir: mjlab's run_train gates those dumps on a `rank` that
    # is hardcoded to 0 on the CPU path, so every worker would write the same
    # two files at the same moment. Checkpoints and wandb are already gated
    # (rsl_rl's Logger keys off gpu_global_rank), so this is the only writer
    # that needs silencing.
    if global_rank_is_nonzero():
        import mjlab.scripts.train as mjlab_train

        mjlab_train.dump_yaml = lambda *args, **kwargs: None

    _patched = True
    print(f"[cpu_distributed] Patch active: gloo process group on CPU ({_rank_env()[0]} workers)")


def global_rank_is_nonzero() -> bool:
    return _rank_env()[2] != 0


def maybe_patch_rsl_rl_for_cpu() -> None:
    """Apply the patch only inside a CPU data-parallel worker."""
    if is_cpu_distributed_worker():
        patch_rsl_rl_for_cpu()
