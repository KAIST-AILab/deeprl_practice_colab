import gym.envs
from rllab.envs.grid_world_env import GridWorldEnv
from envs.proxy_gym_env import ProxyGymEnv
from misc.retro_wrappers import wrap_deepmind_retro
from policies.categorical_mlp_q_policy import CategoricalMlpQPolicy
from exploration_strategies.eps_greedy_strategy import EpsilonGreedyStrategy

from algos.dqn import DQN

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

import lasagne.nonlinearities as NL


def run_task(*_):
    env = GridWorldEnv(desc='chain')
    
    policy = CategoricalMlpQPolicy(
        name='dqn_policy',
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=[],
    )
    
    n_steps = 40000
    es = EpsilonGreedyStrategy(env_spec=env.spec, max_eps=0.5, min_eps=0.05, decay_period=n_steps//2)
    
    algo = DQN(
        env=env,
        policy=policy,
        es=es,
        n_steps=n_steps,
        min_pool_size=50,
        replay_pool_size=100,
        train_epoch_interval=1000,
        max_path_length=50,
        policy_update_method='sgd',
        policy_learning_rate=0.2,
        target_model_update=0.5,
        n_eval_samples=0,
        batch_size=10,

    #     epoch_length=1000,
    #     min_pool_size=10000,
    #     n_epochs=1000,
    #     discount=0.99,
    #     scale_reward=0.01,
    #     qf_learning_rate=1e-3,
    #     policy_learning_rate=1e-4,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()

# run_task()
  
   
run_experiment_lite(
    run_task,
    log_dir='./gridworld_dqn',
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)

