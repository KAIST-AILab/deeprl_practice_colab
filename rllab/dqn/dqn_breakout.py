import gym.envs
from envs.proxy_gym_env import ProxyGymEnv
from misc.retro_wrappers import wrap_deepmind_retro
from policies.categorical_conv_q_policy import CategoricalConvQPolicy
from exploration_strategies.eps_greedy_strategy import EpsilonGreedyStrategy

from algos.dqn import DQN

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

def run_task(*_):
    game = wrap_deepmind_retro(gym.envs.make('BreakoutDeterministic-v4'))

    env = ProxyGymEnv(game, record_video=False, record_log=False)
    
    policy = CategoricalConvQPolicy(
        name='dqn_policy',
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        conv_filters=[32, 64, 64], 
        conv_filter_sizes=[8,4,3], 
        conv_strides=[4,2,1], 
        conv_pads=['valid','valid','valid'],
        hidden_sizes=[512]
    )
    
    n_steps = 10000000 
    es = EpsilonGreedyStrategy(env_spec=env.spec, max_eps=0.5, min_eps=0.05, decay_period=n_steps//2)
    
    algo = DQN(
        env=env,
        policy=policy,
        es=es,
        n_steps=n_steps,
        min_pool_size         =   1000,
        replay_pool_size      =  50000,
        train_epoch_interval  =  10000,
        # max_path_length=np.max,
        policy_update_method='sgd',
        policy_learning_rate=0.005, # needs to be lower...
        target_model_update=0.5,
        n_eval_samples=0,
        batch_size=32,
    #     batch_size=32,
    #     max_path_length=100,
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
    log_dir='./breakout_dqn',
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)

