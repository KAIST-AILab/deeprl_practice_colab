from rllab.algos.trpo import TRPO
# from rllab.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.envs.gym_env import GymEnv

def run_task(*_):
    # Please note that different environments with different action spaces may
    # require different policies. For example with a Discrete action space, a
    # CategoricalMLPPolicy works, but for a Box action space may need to use
    # a GaussianMLPPolicy (see the trpo_gym_pendulum.py example)
    env = normalize(GymEnv(env_name="LunarLanderContinuous-v2", force_reset=True))
    # policy = CategoricalMLPPolicy(
    #     env_spec=env.spec,
    #     # The neural network policy should have two hidden layers, each with 32 hidden units.
    #     hidden_sizes=(32, 32)
    # )
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # max_path_length = env.horizon
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=300,
        n_itr=10000,
        discount=0.99,
        step_size=0.02,
        # truncate_local_is_ratio = 0.2
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
    )
    algo.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    log_dir='./log/trpo_mntcar_cont',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
