from rllab.envs.gym_env import GymEnv
from rllab.algos.ddpg import DDPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction


def run_task(*_):

    env = normalize(GymEnv(env_name = "MountainCarContinuous-v0",force_reset=True))
    max_path_length = 300

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
        batch_size=100,
        n_updates_per_sample = 1,
        max_path_length=max_path_length,
        epoch_length=900,
        min_pool_size=800,
        replay_pool_size = 5000,
        n_epochs=1000,
        discount=0.99,
        scale_reward=0.1,
        qf_learning_rate=1e-3,
        policy_learning_rate=1e-4,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    log_dir='./log/ddpg_mntcar',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
