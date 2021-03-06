from rllab.envs.gym_env import GymEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam
from rllab.misc.instrument import run_experiment_lite
import rllab.misc.logger as logger




def run_task(*_):
    # normalize() makes sure that the actions for the environment lies
    # within the range [-1, 1] (only works for environments with continuous actions)

    env = normalize(GymEnv(env_name = "LunarLanderContinuous-v2", force_reset=True))
    # env = normalize(GymEnv(env_name="CartPole-v0", force_reset=True))

    # Initialize a neural network policy with a single hidden layer of 8 hidden units

    policy = GaussianMLPPolicy(env.spec, hidden_sizes=(64,64))
    # policy = CategoricalMLPPolicy(env.spec, hidden_sizes=(64, 64))

    # We will collect 3 trajectories per iteration
    N = 3
    # Each trajectory will have at most 400 time steps
    T = 400
    # Number of iterations
    n_itr = 1000
    # Set the discount factor for the problem
    discount = 0.99
    # Learning rate for the gradient update
    learning_rate = 0.001

    # Construct the computation graph

    # Create a Theano variable for storing the observations
    # We could have simply written `observations_var = TT.matrix('observations')` instead for this example. However,
    # doing it in a slightly more abstract way allows us to delegate to the environment for handling the correct data
    # type for the variable. For instance, for an environment with discrete observations, we might want to use integer
    # types if the observations are represented as one-hot vectors.
    observations_var = env.observation_space.new_tensor_variable(
        'observations',
        # It should have 1 extra dimension since we want to represent a list of observations
        extra_dims=1
    )
    actions_var = env.action_space.new_tensor_variable(
        'actions',
        extra_dims=1
    )
    returns_var = TT.vector('returns')

    # policy.dist_info_sym returns a dictionary, whose values are symbolic expressions for quantities related to the
    # distribution of the actions. For a Gaussian policy, it contains the mean and the logarithm of the standard deviation.
    dist_info_vars = policy.dist_info_sym(observations_var)

    # policy.distribution returns a distribution object under rllab.distributions. It contains many utilities for computing
    # distribution-related quantities, given the computed dist_info_vars. Below we use dist.log_likelihood_sym to compute
    # the symbolic log-likelihood. For this example, the corresponding distribution is an instance of the class
    # rllab.distributions.DiagonalGaussian
    dist = policy.distribution

    # Note that we negate the objective, since most optimizers assume a minimization problem
    surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * returns_var)

    # Get the list of trainable parameters.
    params = policy.get_params(trainable=True)
    grads = theano.grad(surr, params)

    f_train = theano.function(
        inputs=[observations_var, actions_var, returns_var],
        outputs=None,
        updates=adam(grads, params, learning_rate=learning_rate),
        allow_input_downcast=True
    )

    for epoch in range(n_itr):
        ##################################################################
        logger.push_prefix('Epoch #%d | ' % (epoch))
        logger.log("Training started")
        ##################################################################
        paths = []

        for _ in range(N):
            observations = []
            actions = []
            rewards = []

            observation = env.reset()

            for _ in range(T):
                # policy.get_action() returns a pair of values. The second one returns a dictionary, whose values contains
                # sufficient statistics for the action distribution. It should at least contain entries that would be
                # returned by calling policy.dist_info(), which is the non-symbolic analog of policy.dist_info_sym().
                # Storing these statistics is useful, e.g., when forming importance sampling ratios. In our case it is
                # not needed.
                action, _ = policy.get_action(observation)
                # Recall that the last entry of the tuple stores diagnostic information about the environment. In our
                # case it is not needed.
                next_observation, reward, terminal, _ = env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                observation = next_observation
                if terminal:
                    # Finish rollout if terminal state reached
                    break

            # We need to compute the empirical return for each time step along the
            # trajectory (return to go)
            returns = []
            return_so_far = 0
            for t in range(len(rewards) - 1, -1, -1):
                return_so_far = rewards[t] + discount * return_so_far
                returns.append(return_so_far)
            # The returns are stored backwards in time, so we need to revert it
            returns = returns[::-1]

            paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                returns=np.array(returns)
            ))

        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])

        f_train(observations, actions, returns)
        print('Average Return:', np.mean([sum(p["rewards"]) for p in paths]))
        ############################################################################
        logger.log("Training finished")
        logger.save_itr_params(epoch, params)
        logger.dump_tabular(with_prefix=False)
        logger.pop_prefix()


        logger.record_tabular('Epoch', epoch)
        logger.record_tabular('Steps', epoch*N*T)
        logger.record_tabular('AverageReturn', np.mean(returns))
        logger.record_tabular('StdReturn', np.std(returns))
        logger.record_tabular('MaxReturn', np.max(returns))
        logger.record_tabular('MinReturn', np.min(returns))

        #############################################################################
run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    log_dir='./log/vpg_test',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)