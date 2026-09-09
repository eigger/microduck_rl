import sys, os
from PIL import Image

gif_path = r"C:\Users\eigger\Videos\화면 녹화\녹음 2026-09-09 085735.gif"
if not os.path.exists(gif_path):
    print("GIF not found:", gif_path)
    sys.exit(1)

out_dir = "scratch/user_gif_0857"
os.makedirs(out_dir, exist_ok=True)

im = Image.open(gif_path)
n_frames = getattr(im, "n_frames", 1)
print(f"Total frames: {n_frames}, size: {im.size}")

step = max(1, n_frames // 25)
saved = []
for i in range(0, n_frames, step):
    im.seek(i)
    fname = os.path.join(out_dir, f"frame_{i:03d}.png")
    im.save(fname)
    saved.append(fname)

print(f"Saved {len(saved)} frames to {out_dir}")
