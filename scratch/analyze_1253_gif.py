import os, glob
from PIL import Image
import numpy as np

frames = sorted(glob.glob("scratch/user_gif_1253/frame_*.png"))
print(f"Total extracted frames in scratch/user_gif_1253: {len(frames)}")

# Let's inspect the original gif directly
gif_path = r"C:\Users\eigger\Videos\화면 녹화\녹음 2026-09-09 125347.gif"
if os.path.exists(gif_path):
    im = Image.open(gif_path)
    print(f"Original GIF n_frames: {im.n_frames}, size: {im.size}")
    
    # Save all frames
    os.makedirs("scratch/all_1253_frames", exist_ok=True)
    for i in range(im.n_frames):
        im.seek(i)
        im.save(f"scratch/all_1253_frames/frame_{i:03d}.png")
    print(f"Saved {im.n_frames} frames to scratch/all_1253_frames")
else:
    print(f"File not found: {gif_path}")
