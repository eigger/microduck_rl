import sys, os, glob, math
sys.path.insert(0, 'scripts')
import numpy as np
import mujoco
from PIL import Image
from infer_policy import load_bam_model, load_mujoco_with_bam, PolicyInference, MICRODUCK_XML, BAM_KP_FW, BAM_VIN_MIN

def find_latest_checkpoint():
    runs = sorted(glob.glob("logs/rsl_rl/jump/*"), key=os.path.getmtime, reverse=True)
    for r in runs:
        pts = sorted(glob.glob(os.path.join(r, "model_*.pt")), key=os.path.getmtime, reverse=True)
        if pts:
            return pts[0]
    return None

def run_export_and_eval():
    latest_pt = find_latest_checkpoint()
    print(f"Latest checkpoint found: {latest_pt}")
    if not latest_pt:
        print("No checkpoint found!")
        return

    # Export to policies/jump.onnx if needed
    onnx_path = "policies/jump.onnx"
    if not os.path.exists(onnx_path) or os.path.getmtime(onnx_path) < os.path.getmtime(latest_pt):
        import subprocess
        cmd = [r"C:\Users\eigger\.platformio\penv\Scripts\uv.exe", "run", "python", "scripts/export.py", "Mjlab-Jump-Flat-MicroDuck", "--checkpoint-file", latest_pt, "--onnx-file", onnx_path]
        print(f"Exporting: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("Export failed!")
            return
    else:
        print(f"ONNX policy up to date: {onnx_path}")

    # Evaluate in BAM MuJoCo
    print("\n" + "="*80)
    print("EVALUATING EXPORTED ONNX POLICY")
    print("="*80)

    bam_model = load_bam_model(BAM_KP_FW, vin=7.4, max_current=None)
    model, data, bam_ctrl, _ = load_mujoco_with_bam(
        MICRODUCK_XML, bam_model, 0.005, vin_drop_gain=0.1, vin_min=BAM_VIN_MIN
    )

    policy = PolicyInference(
        model, data, bam_ctrl=bam_ctrl,
        walking_onnx_path='policies/alpha_walking.onnx',
        standing_onnx_path='policies/alpha_stand.onnx',
        jump_onnx_path=onnx_path,
        new_cmd_obs=True, use_projected_gravity=True, jump_duration=0.60,
    )

    renderer = mujoco.Renderer(model, width=640, height=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    camera.distance = 0.85
    camera.elevation = -12.0
    camera.azimuth = 90.0 # Side view for pure vertical height visibility

    mujoco.mj_resetDataKeyframe(model, data, 1)
    mujoco.mj_forward(model, data)

    # Settle in standing
    for _ in range(50):
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

    z_stand = data.qpos[2] * 1000.0
    print(f"Settled Standing Trunk Z: {z_stand:.1f} mm")

    lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    policy.trigger_behavior('jump')

    min_trunk_z = 999.0
    max_trunk_z = 0.0
    max_vz = -999.0
    max_clearance = 0.0
    l_max_clr = 0.0
    r_max_clr = 0.0
    frames = []

    os.makedirs("scratch/jump_eval_frames", exist_ok=True)

    print(f"\n{'Step':<5} | {'t(s)':<5} | {'TrunkZ(mm)':<10} | {'Vz(m/s)':<8} | {'L_Clr(mm)':<9} | {'R_Clr(mm)':<9} | {'MinClr(mm)':<10} | {'Pitch(°)':<8} | {'Contacts':<8}")
    print("-" * 90)

    for step in range(50):
        t = (step + 1) * 0.02
        policy.update_behavior(0.02)
        act = policy.infer()
        policy.apply_action(act)
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        trunk_z = data.qpos[2] * 1000.0
        vz = data.qvel[2]

        l_clr = max(0.0, data.site_xpos[lfoot_id][2] - 0.0147) * 1000.0
        r_clr = max(0.0, data.site_xpos[rfoot_id][2] - 0.0147) * 1000.0
        min_clr = min(l_clr, r_clr)

        quat = data.qpos[3:7]
        w, x, y, z = quat
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))

        lf_con = rf_con = False
        for c in range(data.ncon):
            con = data.contact[c]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ''
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ''
            if 'left_foot' in g1 or 'left_foot' in g2: lf_con = True
            if 'right_foot' in g1 or 'right_foot' in g2: rf_con = True
        con_str = f"{'L' if lf_con else '.'}{'R' if rf_con else '.'}"

        if trunk_z < min_trunk_z: min_trunk_z = trunk_z
        if trunk_z > max_trunk_z: max_trunk_z = trunk_z
        if vz > max_vz: max_vz = vz
        if min_clr > max_clearance: max_clearance = min_clr
        if l_clr > l_max_clr: l_max_clr = l_clr
        if r_clr > r_max_clr: r_max_clr = r_clr

        # Render frame
        renderer.update_scene(data, camera)
        img = Image.fromarray(renderer.render())
        frames.append(img)
        img.save(f"scratch/jump_eval_frames/step_{step:02d}.png")

        print(f"{step:<5} | {t:<5.2f} | {trunk_z:<10.1f} | {vz:<+8.3f} | {l_clr:<9.1f} | {r_clr:<9.1f} | {min_clr:<10.1f} | {pitch:<+8.1f} | {con_str:<8}")

    # Save animated GIF
    gif_path = "scratch/jump_eval_rollout.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=20, loop=0)
    print(f"\nSaved rollout GIF to: {gif_path}")

    print("\n" + "="*80)
    print("FINAL HIGH JUMP MEASUREMENT REPORT")
    print("="*80)
    print(f"Standing Trunk Z:           {z_stand:.1f} mm")
    print(f"Deep Squat Min Trunk Z:     {min_trunk_z:.1f} mm (Squat Depth: {z_stand - min_trunk_z:.1f} mm)")
    print(f"Max Takeoff Vz:             {max_vz:+.3f} m/s")
    print(f"Peak Trunk Height Z:        {max_trunk_z:.1f} mm (Trunk Rise: {max_trunk_z - z_stand:+.1f} mm)")
    print(f"Peak Bilateral Clearance:   {max_clearance:.1f} mm")
    print(f"Left Foot Clearance Peak:   {l_max_clr:.1f} mm")
    print(f"Right Foot Clearance Peak:  {r_max_clr:.1f} mm")
    print("="*80)

if __name__ == '__main__':
    run_export_and_eval()
