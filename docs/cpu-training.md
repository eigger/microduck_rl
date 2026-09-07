# Training without a GPU

mjlab trains through MuJoCo Warp, which wants CUDA. On a machine that has no
CUDA device — an Apple Silicon Mac, say — warp falls back to its CPU device and
everything still runs, just on one core. This is what that costs, what can be
done about it, and what the machine is then actually good for.

All numbers below were measured on an Apple M4 Mac mini (6 performance + 4
efficiency cores, 16 GB) with `Mjlab-Velocity-Flat-MicroDuck`.

## The single-core problem

warp's CPU backend is single-threaded, and there is no knob to widen it —
`wp.config` exposes no thread count at all. A training process sits at ~100%
CPU while the rest of the machine idles:

```
117.9%  101.8%  103.2%  104.2%  101.9%  101.9%     # ps %cpu, 512 envs
```

Almost all of that one core is physics. At 512 envs:

| phase | time | share |
|---|---:|---:|
| Collection (physics, warp) | 14.4 s | 97.8% |
| Learning (PPO update, torch) | 0.31 s | 2.2% |

That ratio decides everything else on this page: the only speedup worth chasing
is more cores collecting.

## The Apple GPU is not an option

Not a missing flag — a missing compiler backend. warp emits code for exactly
two targets: LLVM for CPU, NVRTC for CUDA PTX. Grep warp 1.12.0 for `metal`,
`mps`, `vulkan` or `spir-v` and you get nothing.

```
warp 1.12.0
is_cuda_available  False
devices            ['cpu']
```

`mujoco_warp` is not "MuJoCo with a GPU option" — it is a physics engine
rewritten as warp kernels (28 source files of `@wp.kernel`), so it inherits
warp's target list exactly. CUDA or CPU, nothing between.

torch does see MPS, and it works. But torch only runs the PPO update, which is
that 2.2% above. Moving it to the GPU buys 2% at best; Amdahl's law is
unusually blunt here.

The only path to an Apple GPU would be swapping the simulator for MJX (JAX) and
its experimental `jax-metal` backend. That is a rewrite of everything this repo
builds on top of mjlab — env definitions, DR, BAM actuators, rewards — ending at
a less mature backend. Not recommended.

## Data-parallel across cores

Separate processes do scale: three concurrent 512-env runs cost 16.7 s/iter each
against 14.6 s solo — 3x the throughput for 14% more wall clock. The cores are
genuinely free, so `scripts/train_cpu_dist.py` spends them on ONE run: N workers,
`num_envs` each, gradients averaged over a gloo process group.

```bash
uv run scripts/train_cpu_dist.py --workers 8 Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 192 --agent.max_iterations 1000
```

Everything after the flags goes to `train` untouched. `--base-seed` sets the
seed of worker 0 (each worker runs `base + rank`; identical seeds would give N
copies of one worker's data). `--master-port` pins the rendezvous port.

### num_envs is PER WORKER

The effective PPO batch is `num_envs x workers`. Passing the same
`--env.scene.num-envs` you would use single-process multiplies the batch rather
than splitting the work — halve it yourself to hold the batch fixed. The
alternative, dividing behind the caller's back, makes the config mjlab prints
disagree with what actually ran.

### How it scales

Total envs held at 1536, so every row trains on the same batch:

| workers | envs/worker | iteration | speedup | parallel efficiency |
|---:|---:|---:|---:|---:|
| 1 | 1536 | 45.7 s | 1.00x | 100% |
| 2 | 768 | 24.4 s | 1.87x | 94% |
| 4 | 384 | 14.9 s | 3.06x | 77% |
| 6 | 256 | 11.9 s | 3.83x | 64% |
| 8 | 192 | 10.8 s | 4.23x | 53% |

Two workers are nearly ideal; efficiency falls off after that as each worker's
share of physics shrinks against a fixed synchronisation cost and the cores
start competing for memory bandwidth and boost clocks.

**8 workers beat 6.** The efficiency cores contribute rather than dragging the
synchronous all-reduce, which is the opposite of what you would expect from "the
slowest rank sets the pace" — they simply take on less work and still help. Past
6 the returns flatten, so 6-8 is the useful range.

Each worker builds its own MuJoCo world: budget ~1.2-1.4 GB of RSS per worker at
512 envs, which is what caps worker count on a 16 GB machine well before core
count does.

## What is patched, and why it survives an upstream merge

rsl_rl's distributed path is written for CUDA and only for CUDA
(`rsl_rl/runners/on_policy_runner.py::_configure_multi_gpu`):

1. `if self.device != f"cuda:{local_rank}": raise ValueError` — "cpu" is
   rejected before anything else happens.
2. `init_process_group(backend="nccl")` — nccl has no macOS build.
3. `torch.cuda.set_device(local_rank)` — no device to set.

Those three lines are the entire blocker. Every collective in `PPO`
(`all_reduce`, `broadcast`, `broadcast_object_list`) is already
backend-agnostic, and gloo implements all of them on CPU tensors.

`src/mjlab_microduck/cpu_distributed.py` replaces that one method and nothing
else. `WORLD_SIZE=1` and any CUDA device delegate to the original, so
single-process training and real multi-GPU boxes behave exactly as before. It is
applied from `tasks/__init__.py` — the same plugin-loader import path the
existing rsl_rl patches in `mdp.py` use — so workers pick it up without being
told.

No upstream file is forked or vendored. The whole footprint inside files this
repo shares with upstream is eight lines in `tasks/__init__.py`;
`tests/test_cpu_distributed.py` locks the property that matters most, which is
the one that is invisible when it holds: the single-process and CUDA paths must
keep reaching upstream's own code.

Workers past rank 0 also get `dump_yaml` silenced, because mjlab's `run_train`
hardcodes `rank = 0` on the CPU path and every worker would otherwise race to
write the same `params/*.yaml`. Checkpoints and wandb are already gated on
rsl_rl's `gpu_global_rank`.

## So what is this machine for

**Smoke tests — yes, and this is the real win.** 64 envs x 5 iterations takes
**13 seconds** single-process. AGENTS.md asks for one before every long run, and
now it costs nothing and needs no GPU. Add `WANDB_MODE=offline` and
`--gpu-ids None`:

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
    --env.scene.num-envs 64 --agent.max_iterations 5 --gpu-ids None
```

**Short experiments and seed sweeps — workable.** Either 8-way data-parallel, or
several independent single-process runs (each is single-threaded, so 6-8 of them
coexist happily). For comparing reward weights or seeds, independent runs give
more information per core-hour than one faster run does.

**Real training — no.** A gait needs 4000-6000 iterations at 4096 envs. Even at
the measured 4.23x that is well over a day of the machine being unusable for
anything else, against an hour or two on one datacenter GPU. Use `--hf-jobs`
(see `scripts/hf/README.md`), which exists for exactly this.
