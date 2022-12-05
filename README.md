# Introduction
This repo is for MuJoCo learning and PPO algorithm testing.

----
# Description

**/assets** directory includes all robots models in .urdf, .xml, .mjb, .txt formats.

**/config** directory include all environment configurations and settings .yml files for training, located at **/config/cfg**. Besides, **parse_args.py** parses commands from terminal, **config.py** loads .yml files.

**/utils** includes useful modules but less related to RL core algorithm.

**/lib** includes all RL related core modules.

**/tmp** is the logging file directory.

**/structural_control** is the customized agent and env that inherits from lib.

> Please note that: results are generated in **/tmp** in default, but well-trained results will be moved to **/results** manually.
----------------------

# Usage
Training:
```commandline
python train.py --domain pendulum --task swingup --num_threads 10
```
Evaluation:
```commandline
python eval.py --domain pendulum --task swingup --rec 20221205_110244 --iter best
```
----------------------
# Environment
- python==3.9
- mujoco==2.3.0
- dm_control==1.0.8
- glfw==2.5.5
- numpy==1.23.3
- gym==0.26.2
- tensorboard==2.11.0
- protobuf==3.20.3
### mac m1
- torch==1.12.1
- torchvision==0.13.1
- torch-geometric==2.1.0
- torch-scatter==2.0.9
- torch-sparse==0.6.15
- torch-cluster==1.6.0
- torch-spline-conv==1.2.1
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