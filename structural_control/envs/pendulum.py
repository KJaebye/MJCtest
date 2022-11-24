# ------------------------------------------------------------------------------------------------------------------- #
#   @description: Class PendulumEnv, PendulumPhysics, SwingUpTask
#   @author: by Kangyao Huang
#   @created date: 24.Nov.2022
# ------------------------------------------------------------------------------------------------------------------- #

import collections

from lib.envs.mujoco_env import MujocoEnv, MujocoPhysics, MujocoTask
from dm_control.utils import rewards
import dm_env
import numpy as np

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))


class PendulumEnv(MujocoEnv):
    def __init__(self, cfg, **kwargs):
        self.mujoco_xml_path = './assets/robot_models/mjcf/pendulum.xml'
        # self.mujoco_xml_path = '/Users/kjaebye/EvoTest/MJCtest/assets/robot_models/mjcf/pendulum.xml'
        self.cfg = cfg
        physics = PendulumPhysics.from_xml_path(self.mujoco_xml_path)
        task = SwingUpTask(random=None)
        super().__init__(physics, task, time_limit=_DEFAULT_TIME_LIMIT, **kwargs)


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
        # done = np.isfinite(s).all() and (self._step_count < max_nsteps)
        done = False
        return done

    def state_vector(self):
        return np.concatenate([
            self.physics.data.qpos.flat,
            self.physics.data.qvel.flat
        ])


class PendulumPhysics(MujocoPhysics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_vertical(self):
        """Returns vertical (z) component of pole frame."""
        return self.named.data.xmat['pole', 'zz']

    def angular_velocity(self):
        """Returns the angular velocity of the pole."""
        return self.named.data.qvel['hinge'].copy()

    def pole_orientation(self):
        """Returns both horizontal and vertical components of pole frame."""
        return self.named.data.xmat['pole', ['zz', 'xz']]


class SwingUpTask(MujocoTask):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Pendulum`.

        Args:
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to a random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
        physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating pole orientation and angular velocity
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Pendulum physics.

        Returns:
          A `dict` of observation.
        """
        obs = collections.OrderedDict()
        obs['orientation'] = physics.pole_orientation()
        obs['velocity'] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))
