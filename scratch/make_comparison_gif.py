from PIL import Image, ImageDraw, ImageFont
import glob, os

im1 = Image.open("scratch/run26_sideview.gif")
im2 = Image.open("scratch/run30_sideview.gif")

n_frames = min(im1.n_frames, im2.n_frames)
w, h = im1.size

combined_frames = []

for i in range(n_frames):
    im1.seek(i)
    im2.seek(i)
    
    f1 = im1.convert('RGB')
    f2 = im2.convert('RGB')
    
    combined = Image.new('RGB', (w * 2 + 10, h + 40), (30, 30, 35))
    combined.paste(f1, (0, 40))
    combined.paste(f2, (w + 10, 40))
    
    draw = ImageDraw.Draw(combined)
    draw.text((w // 2 - 80, 10), "Run 26 (Peak 146mm, 5.4cm tuck, head drop)", fill=(255, 200, 100))
    draw.text((w + 10 + w // 2 - 80, 10), "Run 30 (Upright head, but legs straight)", fill=(100, 220, 255))
    draw.text((w // 2 - 40, h + 15), f"Frame {i:02d}", fill=(200, 200, 200))
    
    combined_frames.append(combined)

combined_frames[0].save("scratch/comparison.gif", save_all=True, append_images=combined_frames[1:], duration=20, loop=0)
combined_frames[15].save("scratch/comparison_frame_15.png")
print("Saved scratch/comparison.gif and scratch/comparison_frame_15.png!")
