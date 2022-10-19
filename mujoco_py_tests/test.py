import mujoco_py
import copy

# load model from xml file
# hopper_path = '../assets/robot_models/mjcf/hopper.xml'
# ant_path = '../assets/robot_models/mjcf/ant.xml'
# gap_path = '../assets/robot_models/mjcf/gap.xml' # load failed since this xml requires the definition of terrain
# swimmer_path = '../assets/robot_models/mjcf/swimmer.xml'
pendulum_path = '../assets/robot_models/mjcf/pendulume.xml'
# basic_example_path = '../assets/robot_models/mjcf/basic_example.xml'

# hopper_model = mujoco_py.load_model_from_path(hopper_path)
# ant_model = mujoco_py.load_model_from_path(ant_path)
# gap_model = mujoco_py.load_model_from_path(gap_path)
# swimmer_model = mujoco_py.load_model_from_path(swimmer_path)
pendulume_model = mujoco_py.load_model_from_path(pendulum_path)
# basic_example_model = mujoco_py.load_model_from_path(basic_example_path)

# load from xml
# self-defined model
# self_defined_xml = """
# <mujoco>
#    <worldbody>
#       <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
#       <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
#       <body pos="0 0 1">
#          <joint type="free"/>
#          <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
#       </body>
#    </worldbody>
# </mujoco>
#
# """
# self_defined_model = mujoco_py.load_model_from_xml(self_defined_xml)

# model
model = pendulume_model
# model = basic_example_model

# create simulation
sim = mujoco_py.MjSim(model, data=None, nsubsteps=1, udd_callback=None)
# create a viewer
viewer = mujoco_py.MjViewer(sim)

# definition
t = 0

while True:
    viewer.render()
    t += 1
    sim.step()
    if t >= 100000000:
        break