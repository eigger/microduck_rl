from PIL import Image
import os

im = Image.open("scratch/run26_sideview.gif")
os.makedirs("scratch/run26_frames", exist_ok=True)
for i in range(im.n_frames):
    im.seek(i)
    im.save(f"scratch/run26_frames/frame_{i:02d}.png")
print("Done extracting run26 frames!")
