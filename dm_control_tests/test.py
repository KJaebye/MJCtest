# import mujoco_env
import sys
sys.path.append('..')
import lib.envs.mujoco_env as mujoco_env
import collections
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
from dm_control import viewer

_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


class HopperEnv(mujoco_env.MujocoEnv):
    def __init__(self, cfg):
        physics = HopperPhysics.from_xml_path('hopper.xml')
        task = HopperTask(True, random=None)
        super().__init__(cfg, physics, task)


class HopperPhysics(mujoco_env.MujocoPhysics):
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


class HopperTask(mujoco_env.MujocoTask):
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
        super().__init__(random=random)

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


env = HopperEnv(None)
data = env.physics.data
action_spec = env.action_spec()
observation_spec = env.observation_spec()
print(len(observation_spec))


# Define a uniform random policy.
def random_policy(time_step):
    del time_step  # Unused.
    return np.random.uniform(low=action_spec.minimum,
                             high=action_spec.maximum,
                             size=action_spec.shape)


viewer.launch(env, policy=random_policy)
