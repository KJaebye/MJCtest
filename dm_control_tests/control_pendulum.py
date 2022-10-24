from dm_control import mujoco
from dm_control import _render
from dm_control import suite

simple_MJCF = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

physics = mujoco.Physics.from_xml_string(simple_MJCF)

glfw.init()
window = glfw.create_window(1200, 900, "test", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

ctx = mujoco.GLContext(1200, 900)
ctx.make_current()


duration = 10
physics.reset()

while not glfw.window_should_close(window):
    start = physics.time()

    while physics.time() - start < 1.0/60.0:
        physics.step()

    if physics.time() >= duration:
        break

    ctx.make_current()
    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
ctx.free()

# # Load an environment from the Control Suite.
# env = suite.load(domain_name="humanoid", task_name="stand")
#
# # Launch the viewer application.
# viewer.launch(env)