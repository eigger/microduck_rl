import mujoco
from PIL import Image

m = mujoco.MjModel.from_xml_path("src/mjlab_microduck/robot/microduck/scene.xml")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 1)
mujoco.mj_forward(m, d)

renderer = mujoco.Renderer(m, 480, 640)
renderer.update_scene(d)
img = renderer.render()
print("Rendered shape:", img.shape)
Image.fromarray(img).save("scratch/test_render.png")
print("Saved scratch/test_render.png")
