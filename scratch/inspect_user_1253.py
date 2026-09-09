import os
from PIL import Image, ImageChops

files = sorted([f for f in os.listdir('scratch/user_gif_1253') if f.endswith('.png')])
for i in range(1, len(files)):
    im1 = Image.open('scratch/user_gif_1253/' + files[i-1]).convert('L')
    im2 = Image.open('scratch/user_gif_1253/' + files[i]).convert('L')
    diff = sum(ImageChops.difference(im1, im2).getdata()) // 1000
    if diff > 10:
        print(f"{files[i]}: diff={diff}")
