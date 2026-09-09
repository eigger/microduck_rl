from PIL import Image
im = Image.open("scratch/run26_rollout.gif")
for i in [0, 5, 10, 15, 20, 45, 50, 57]:
    im.seek(i)
    im.save(f"scratch/sim_frame_{i:02d}.png")
print("Saved sim frames")
