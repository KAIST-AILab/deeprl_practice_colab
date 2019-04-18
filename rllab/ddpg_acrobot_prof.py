from rllab.envs.gym_env import GymEnv
from rllab.algos.ddpg import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


def run_task(*_):
    # env = normalize(HalfCheetahEnv())

    env = normalize(GymEnv(env_name = "LunarLanderContinuous-v2"))
    # env = normalize(GymEnv(env_name="BipedalWalker-v2", force_reset=True, record_video=True))
    max_path_length = 400
    # print("env.horizon: ",env.horizon)
    # input()
    # env._max_episode_steps = max_path_length

    policy = DeterministicMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers
        hidden_sizes=(64, 64)
    )


    es = OUStrategy(env_spec=env.spec)

    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=(64, 64)
                                )

    algo = DDPG(
        env=env,
        policy=policy,
        es=es,
        qf=qf,
        batch_size=64,
        max_path_length=max_path_length,
        epoch_length=1000,
        min_pool_size=900,
        replay_pool_size = 10000,
        n_updates_per_sample =1,
        n_epochs = 3000,
        discount=0.99,
        scale_reward=0.1,
        qf_learning_rate=1e-2,
        policy_learning_rate=5e-3,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    log_dir='./log/ddpg_mntcar_cont',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
