#!/usr/bin/env python3
"""
身体如何相互作用的例子。 为了身体能够移动它需要有关节。 在这个例子中，“机器人”是一个红球
带有 X 和 Y 滑动接头（以及不受控制的 Z 滑动接头）。
在地板上，有一个带有 X 和 Y 滑动接头的圆柱体，所以它可以被机器人推来推去。 还有一个没有接头的盒子。 由于
盒子没有关节，是固定的，不能推来推去。
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="robot" pos="0 0 1.2">
            <joint axis="1 0 0" damping="0.1" name="slide0" pos="0 0 0" type="slide"/>
            <joint axis="0 1 0" damping="0.1" name="slide1" pos="0 0 0" type="slide"/>
            <joint axis="0 0 1" damping="1" name="slide2" pos="0 0 0" type="slide"/>
            <geom mass="1.0" pos="0 0 0" rgba="1 0 0 1" size="0.15" type="sphere"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        </body>
        <body mocap="true" name="mocap" pos="0.5 0.5 0.5">
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.1 0.1 0.1" type="box"></geom>
			<geom conaffinity="0" contype="0" pos="0 0 0" rgba="1.0 1.0 1.0 0.5" size="0.2 0.2 0.05" type="box"></geom>
		</body>
        <body name="cylinder" pos="0.1 0.1 0.2">
            <geom mass="1" size="0.15 0.15" type="cylinder"/>
            <joint axis="1 0 0" name="cylinder:slidex" type="slide"/>
            <joint axis="0 1 0" name="cylinder:slidey" type="slide"/>
        </body>
        <body name="box" pos="-0.8 0 0.2">
            <geom mass="0.1" size="0.15 0.15 0.15" type="box"/>
        </body>
        <body name="floor" pos="0 0 0.025">
            <geom condim="3" size="1.0 1.0 0.02" rgba="0 1 0 1" type="box"/>
        </body>
    </worldbody>
    <actuator>
        <motor gear="2000.0" joint="slide0"/>
        <motor gear="2000.0" joint="slide1"/>
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)  # 从包含 XML标记的字符串加载并返回 PyMjModel模型。
sim = MjSim(model) # MjSim 表示一个正在运行的模拟，包括其状态。
viewer = MjViewer(sim) # 要显示的模拟器。
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step() #   通过调用`mj_step`来推进仿真。
    viewer.render() # 从摄像机渲染视图，并将图像作为 numpy.ndarray 返回。
    if t > 100 and os.getenv('TESTING') is not None:
        break

# getenv():获取环境变量，如果不存在则返回None。
