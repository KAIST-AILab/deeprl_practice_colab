from rllab.core.lasagne_powered import LasagnePowered
import lasagne.layers as L
from rllab.core.network import ConvNetwork
from rllab.distributions.categorical import Categorical
from rllab.policies.base import Policy
from rllab.misc import tensor_utils
from rllab.spaces.discrete import Discrete

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
import numpy as np
import lasagne.nonlinearities as NL


class CategoricalConvQPolicy(Policy, LasagnePowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            conv_filters, conv_filter_sizes, conv_strides, conv_pads,
            hidden_sizes=[],
            hidden_nonlinearity=NL.rectify
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        self._env_spec = env_spec
        
        # print( env_spec.observation_space.shape )

        q_network = ConvNetwork(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.linear,
            name=name,
        )

        self._l_q = q_network.output_layer
        self._l_obs = q_network.input_layer
        self._f_q = ext.compile_function(
            [q_network.input_layer.input_var],
            L.get_output(q_network.output_layer)
        )

        self._dist = Categorical(env_spec.action_space.n)

        super(CategoricalConvQPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [q_network.output_layer])

    @property
    def vectorized(self):
        return True

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        q = self._f_q([flat_obs])[0]
        action = np.argmax(q)
        return action, dict(q=q)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        q = self._f_q(flat_obs)
        actions = np.argmax(q, axis=1)
        return actions, dict(q=q)
    
        
    def get_qval_sym(self, obs_var, a_var):
        q_vals_sym = L.get_output(self._l_q, 
                                  {self._l_obs: obs_var})
        
        return (q_vals_sym * a_var).sum(axis=1)
    
    def get_qval_sym_test(self, obs_var, a_var):
        q_vals_sym = L.get_output(self._l_q, 
                                  {self._l_obs: obs_var})
        
        return [q_vals_sym, a_var, q_vals_sym * a_var, (q_vals_sym * a_var).sum(axis=1)]
