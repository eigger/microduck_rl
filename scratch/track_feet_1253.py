import glob, os
from PIL import Image
import numpy as np

# Let's inspect frames at interval of 10: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
for i in range(0, 108, 10):
    fn = f"scratch/all_1253_frames/frame_{i:03d}.png"
    im = Image.open(fn).convert('RGB')
    arr = np.array(im)
    # The floor is blue/checkerboard, the robot is white/orange
    # Orange soles: R > 200, G ~ 100, B < 50
    orange_mask = (arr[:, :, 0] > 180) & (arr[:, :, 1] > 80) & (arr[:, :, 1] < 160) & (arr[:, :, 2] < 60)
    orange_ys, orange_xs = np.where(orange_mask)
    if len(orange_ys) > 0:
        min_y = np.min(orange_ys)
        max_y = np.max(orange_ys)
        print(f"Frame {i:03d}: Orange feet y range: [{min_y}, {max_y}] (Height in image: {max_y - min_y} px)")
    else:
        print(f"Frame {i:03d}: No feet detected")
