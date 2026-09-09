import os
import sys
from PIL import Image

gif_path = r"C:\Users\eigger\Videos\화면 녹화\녹음 2026-09-09 083023.gif"
out_dir = r"scratch\user_gif_0830"
os.makedirs(out_dir, exist_ok=True)

im = Image.open(gif_path)
n_frames = im.n_frames
print(f"Total frames: {n_frames}")
step = max(1, n_frames // 15)
for i in range(0, n_frames, step):
    im.seek(i)
    im.save(os.path.join(out_dir, f"frame_{i:03d}.png"))
print("Extracted frames!")
