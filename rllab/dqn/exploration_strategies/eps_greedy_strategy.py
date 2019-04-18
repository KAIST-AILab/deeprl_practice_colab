from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np


class EpsilonGreedyStrategy(ExplorationStrategy, Serializable):
    """
    This strategy adds Epsilon noise to the action taken by the deterministic policy.
    """

    def __init__(self, env_spec, max_eps=1.0, min_eps=0.1, decay_period=10000000):
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay_period = decay_period
        self._env_spec = env_spec
        self._action_space = env_spec.action_space

    def get_action(self, t, observation, policy, **kwargs):
        action, agent_info = policy.get_action(observation)
        
#         print(self._env_spec.observation_space.unflatten(observation), 
#               self._env_spec.action_space.unflatten(action))
        eps = self._max_eps - (self._max_eps - self._min_eps) * min(1.0, t * 1.0 / self._decay_period)
        # print(t, eps, observation, action, agent_info)
        
        if np.random.uniform() < eps:
            action = np.random.randint(self._action_space.n)
            
        return action
        