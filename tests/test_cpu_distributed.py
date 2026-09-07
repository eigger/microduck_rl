"""The CPU data-parallel patch must not touch the single-process path.

The patch rewrites an rsl_rl method every training run imports, so the property
that matters most is the one that is invisible when it holds: a normal
single-process run, and any run on real GPUs, must behave exactly as upstream.
These lock that, plus the three CPU-path substitutions the patch exists to make
(gloo not nccl, "cpu" accepted as a device, no torch.cuda.set_device).
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from mjlab_microduck import cpu_distributed


@pytest.fixture(autouse=True)
def _clean_patch_state():
    """Each test patches from scratch: the module memoizes with a global."""
    before = cpu_distributed._patched
    cpu_distributed._patched = False
    yield
    cpu_distributed._patched = before


def _env(**kwargs) -> dict[str, str]:
    base = {"WORLD_SIZE": "1", "LOCAL_RANK": "0", "RANK": "0", "CUDA_VISIBLE_DEVICES": ""}
    base.update({k: str(v) for k, v in kwargs.items()})
    return base


class _Runner:
    """Stand-in for OnPolicyRunner: only what _configure_multi_gpu touches."""

    def __init__(self, device: str):
        self.device = device
        self.cfg: dict = {}


def test_worker_detection_requires_both_world_size_and_no_cuda():
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=1), clear=True):
        assert not cpu_distributed.is_cpu_distributed_worker()
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=4), clear=True):
        assert cpu_distributed.is_cpu_distributed_worker()
    # A real multi-GPU run must NOT be treated as the CPU case.
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=4, CUDA_VISIBLE_DEVICES="0,1"), clear=True):
        assert not cpu_distributed.is_cpu_distributed_worker()


def test_maybe_patch_is_a_noop_outside_a_worker():
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=1), clear=True):
        cpu_distributed.maybe_patch_rsl_rl_for_cpu()
    assert OnPolicyRunner._configure_multi_gpu is original


def test_single_process_delegates_to_upstream():
    """WORLD_SIZE=1 must reach upstream's own code, not the CPU branch."""
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=4), clear=True):
        cpu_distributed.patch_rsl_rl_for_cpu()
    patched = OnPolicyRunner._configure_multi_gpu
    try:
        assert patched is not original
        runner = _Runner("cpu")
        with mock.patch.dict(os.environ, _env(WORLD_SIZE=1), clear=True):
            patched(runner)
        # Upstream's not-distributed result, untouched.
        assert runner.is_distributed is False
        assert runner.cfg["multi_gpu"] is None
        assert runner.gpu_global_rank == 0
    finally:
        OnPolicyRunner._configure_multi_gpu = original


def test_cuda_device_still_delegates_to_upstream():
    """A GPU box must keep upstream's nccl path, patch installed or not.

    Upstream rejects a device that does not match its local rank; seeing that
    exact error for "cuda:1" at LOCAL_RANK=0 proves the call reached upstream
    rather than our CPU branch, which has no such check.
    """
    import torch
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=2), clear=True):
        cpu_distributed.patch_rsl_rl_for_cpu()
    patched = OnPolicyRunner._configure_multi_gpu
    try:
        with mock.patch.dict(
            os.environ,
            _env(WORLD_SIZE=2, LOCAL_RANK=0, RANK=0, CUDA_VISIBLE_DEVICES="0,1"),
            clear=True,
        ):
            with mock.patch.object(torch.distributed, "init_process_group") as init:
                with pytest.raises(ValueError, match="does not match expected device"):
                    patched(_Runner("cuda:1"))
        init.assert_not_called()
    finally:
        OnPolicyRunner._configure_multi_gpu = original


def test_cpu_path_uses_gloo_and_never_touches_cuda():
    """The three substitutions the patch exists to make."""
    import torch
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=3), clear=True):
        cpu_distributed.patch_rsl_rl_for_cpu()
    patched = OnPolicyRunner._configure_multi_gpu
    try:
        runner = _Runner("cpu")
        with mock.patch.dict(
            os.environ, _env(WORLD_SIZE=3, LOCAL_RANK=2, RANK=2), clear=True
        ):
            with mock.patch.object(torch.distributed, "init_process_group") as init:
                with mock.patch.object(torch.distributed, "is_initialized", return_value=False):
                    with mock.patch.object(torch.cuda, "set_device") as set_device:
                        patched(runner)

        # 1. device "cpu" accepted rather than raising on the cuda:N compare.
        assert runner.is_distributed is True
        assert runner.gpu_world_size == 3
        assert runner.gpu_local_rank == 2
        assert runner.gpu_global_rank == 2
        assert runner.cfg["multi_gpu"] == {
            "global_rank": 2,
            "local_rank": 2,
            "world_size": 3,
        }
        # 2. gloo, because nccl has no macOS build.
        init.assert_called_once()
        assert init.call_args.kwargs["backend"] == "gloo"
        assert init.call_args.kwargs["world_size"] == 3
        assert init.call_args.kwargs["rank"] == 2
        # 3. no CUDA device to pin.
        set_device.assert_not_called()
    finally:
        OnPolicyRunner._configure_multi_gpu = original


def test_cpu_path_rejects_rank_beyond_world_size():
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=2), clear=True):
        cpu_distributed.patch_rsl_rl_for_cpu()
    patched = OnPolicyRunner._configure_multi_gpu
    try:
        with mock.patch.dict(
            os.environ, _env(WORLD_SIZE=2, LOCAL_RANK=5, RANK=5), clear=True
        ):
            with pytest.raises(ValueError, match="rank out of range"):
                patched(_Runner("cpu"))
    finally:
        OnPolicyRunner._configure_multi_gpu = original


def test_patch_is_idempotent():
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    original = OnPolicyRunner._configure_multi_gpu
    with mock.patch.dict(os.environ, _env(WORLD_SIZE=2), clear=True):
        cpu_distributed.patch_rsl_rl_for_cpu()
        first = OnPolicyRunner._configure_multi_gpu
        cpu_distributed.patch_rsl_rl_for_cpu()
        second = OnPolicyRunner._configure_multi_gpu
    try:
        assert first is second
    finally:
        OnPolicyRunner._configure_multi_gpu = original
