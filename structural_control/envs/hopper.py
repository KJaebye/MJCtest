# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class HopperEnv, HopperPhysics, HopperTask
#   @author: by Kangyao Huang
#   @created date: 07.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

from abc import ABC
import numpy as np
import collections
import dm_env
from dm_control.utils import rewards
from dm_control.suite.utils import randomizers
from lib.envs.mujoco_env import MujocoEnv, MujocoTask, MujocoPhysics

# Default simulation timestep is 0.005s in hopper.xml
# Default control timestep
_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


class HopperEnv(MujocoEnv):
    def __init__(self, cfg, **kwargs):
        self.mujoco_xml_path = './assets/robot_models/mjcf/hopper.xml'
        self.cfg = cfg
        physics = HopperPhysics.from_xml_path(self.mujoco_xml_path)
        task = HopperTask(hopping=False, random=None)
        super().__init__(physics, task, time_limit=_DEFAULT_TIME_LIMIT, control_timestep=_CONTROL_TIMESTEP, **kwargs)

    def step(self, action):
        """Updates the environment using the action and returns a `TimeStep`."""
        # apply action
        self._task.before_step(action, self._physics)
        self._physics.step(self._n_sub_steps)
        self._task.after_step(self._physics)

        # observation
        observation = self._task.get_observation(self._physics)
        # reward
        reward = self._task.get_reward(self._physics)
        # step
        self._step_count += 1

        done = self.check_done(observation)

        if self._step_count >= self._step_limit:
            discount = 1.0
        else:
            discount = self._task.get_termination(self._physics)

        episode_over = discount is not None
        if episode_over:
            self._reset_next_step = True
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount, observation), done
        else:
            return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation), done

    def check_done(self, observation):
        """ check agent is well done """
        s = self.state_vector()
        max_nsteps = 300
        done = np.isfinite(s).all() and (self.physics.height() > _STAND_HEIGHT) \
               and (self._step_count < max_nsteps) and (not self.physics.touch().any())
        return done

    def state_vector(self):
        return np.concatenate([
            self.physics.data.qpos.flat,
            self.physics.data.qvel.flat
        ])


class HopperPhysics(MujocoPhysics):
    def height(self):
        """Returns height of torso with respect to foot."""
        return (self.named.data.xipos['torso', 'z'] -
                self.named.data.xipos['foot', 'z'])

    def speed(self):
        """Returns horizontal speed of the Hopper."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def touch(self):
        """Returns the signals from two foot touch sensors."""
        return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])


class HopperTask(MujocoTask, ABC):
    def __init__(self, hopping, random=None):
        """Initialize an instance of `Hopper`.

        Args:
          hopping: Boolean, if True the task is to hop forwards, otherwise it is to
            balance upright.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._hopping = hopping
        super().__init__()

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of positions, velocities and touch sensors."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance:
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        obs['touch'] = physics.touch()
        return obs

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        if self._hopping:
            hopping = rewards.tolerance(physics.speed(),
                                        bounds=(_HOP_SPEED, float('inf')),
                                        margin=_HOP_SPEED / 2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')
            return standing * hopping
        else:
            small_control = rewards.tolerance(physics.control(),
                                              margin=1, value_at_margin=0,
                                              sigmoid='quadratic').mean()
            small_control = (small_control + 4) / 5
            return standing * small_control
