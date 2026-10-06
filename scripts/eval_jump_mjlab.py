"""Headless jump rollouts of an exported ONNX in the training env (mjlab).

Counterpart to eval_run90_squat_jump_cushion_stand.py (CPU MuJoCo + BAM): if a
policy stands here and falls there, the problem is sim2sim, not the reward.

    python scripts/eval_jump_mjlab.py scratch/policy.onnx [--num-envs 256] [--play]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import make_microduck_jump_env_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--play", action="store_true", help="play cfg (no DR / noise)")
    ap.add_argument("--no-stand-curriculum", action="store_true")
    args = ap.parse_args()

    cfg = make_microduck_jump_env_cfg(play=args.play)
    cfg.scene.num_envs = args.num_envs
    if args.no_stand_curriculum and "reset_stand_curriculum" in cfg.events:
        cfg.events["reset_stand_curriculum"].params["fraction"] = 0.0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset(seed=0)
    robot = env.scene["robot"]

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    batched = sess.get_inputs()[0].shape[0] != 1

    def act(o: torch.Tensor) -> torch.Tensor:
        x = o.detach().cpu().numpy().astype(np.float32)
        if batched:
            a = sess.run(None, {in_name: x})[0]
        else:
            a = np.concatenate([sess.run(None, {in_name: x[i : i + 1]})[0] for i in range(len(x))])
        return torch.as_tensor(a, device=device)

    n = args.num_envs
    start_xy = robot.data.root_link_pos_w[:, :2].clone()
    start_yaw = microduck_mdp.yaw_from_quat(robot.data.root_link_quat_w).clone()
    stand = getattr(env, "_stand_now", torch.zeros(n, device=device)).clone() > 0.5
    done = torch.zeros(n, dtype=torch.bool, device=device)
    fell = torch.zeros(n, dtype=torch.bool, device=device)
    max_tilt = torch.zeros(n, device=device)
    max_z = torch.zeros(n, device=device)
    final = {}
    steps = int(round(cfg.episode_length_s / (cfg.sim.mujoco.timestep * cfg.decimation)))
    for t in range(steps):
        obs, _, term, trunc, _ = env.step(act(obs["actor"]))
        alive = ~done
        q = robot.data.root_link_quat_w
        cos_tilt = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
        tilt = torch.rad2deg(torch.acos(cos_tilt.clamp(-1, 1)))
        z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
        max_tilt = torch.where(alive, torch.maximum(max_tilt, tilt), max_tilt)
        max_z = torch.where(alive, torch.maximum(max_z, z), max_z)
        fell |= alive & term
        if t == steps - 2:
            fwd, left = microduck_mdp.spawn_frame_offset(
                start_xy, start_yaw, robot.data.root_link_pos_w[:, :2]
            )
            yaw = microduck_mdp.yaw_from_quat(q) - start_yaw
            final = {
                "fwd_mm": fwd * 1000,
                "left_mm": left * 1000,
                "yaw_deg": torch.rad2deg(torch.atan2(torch.sin(yaw), torch.cos(yaw))),
                "z_mm": z * 1000,
                "tilt_deg": tilt,
            }
        done |= term | trunc

    def summary(mask: torch.Tensor, label: str):
        k = int(mask.sum())
        if k == 0:
            return
        ok = mask & ~fell
        print(f"\n[{label}] n={k}  fell={int((mask & fell).sum())} ({100*float((mask & fell).sum())/k:.1f}%)")
        print(f"  max tilt  median {float(max_tilt[mask].median()):.1f}°  p90 {float(max_tilt[mask].quantile(0.9)):.1f}°")
        print(f"  max z     median {float(max_z[mask].median())*1000:.1f} mm")
        if int(ok.sum()):
            for key, v in final.items():
                x = v[ok]
                print(f"  {key:9s} median {float(x.median()):+7.1f}  p10 {float(x.quantile(0.1)):+7.1f}  p90 {float(x.quantile(0.9)):+7.1f}")

    summary(~stand, "squat spawns")
    summary(stand, "stand-curriculum spawns")


if __name__ == "__main__":
    main()
