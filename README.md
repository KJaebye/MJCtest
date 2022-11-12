# Introduction
This repo is for MuJoCo learning and testing (based on the architecture of **Transform2Act** and **NerveNet** works)


Learning records and tips are located at Notion https://www.notion.so/huangkangyao/Transform2Act-Reconstruction-6edeb95a2aea449fa016282a7f990e22

----
# Description

**/mujoco_native_tests** directory provides examples using **MuJoCo official python bindings** API.

**/mujoco_py_tests** directory provides examples that use the third-party python binding **mujoco-py** API. (Unmaintained, maintained by OpenAI before acquisition)

**/dm_control_tests** directory provides examples using dm_control.

---

**/assets** directory includes all robots models in .urdf, .xml, .mjb, .txt formats.

**/config** directory include all environment configurations and settings .yml files for training, located at **/config/cfg**. Besides, **parse_args.py** parses commands from terminal, **config.py** loads .yml files.

**/utils** includes useful modules but less related to RL core algorithm.

**/lib** includes all RL related core modules.

**/tmp** is the logging file directory.


> Please note that: results are generated in **/tmp** in default, but well-trained results will be moved to **/results** manually.
----------------------
# Environment
- python==3.9
- mujoco==2.3.0
- dm_control==1.0.8
- glfw==2.5.5
- mujoco-python-viewer==0.1.2
- numpy==1.23.3
- gym==0.26.2
- tensorboard==2.11.0
- protobuf==3.20.3
### mac m1
- torch==1.12.1
- torchvision==0.13.1
### x_86
- torch==1.12.0_cu116
- torchvision==0.13.0_cu116
- torch-cluster==1.6.0
- torch-geometric==2.1.0
- torch-scatter==2.0.9
- torch-sparse==0.6.14
- torch-spline-conv==1.2.1
- gym==0.15.4
- mujoco-py==2.1.2.14