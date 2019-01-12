from rllab.envs.gym_env import GymEnv
from rllab.algos.vpg import VPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
# from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer #sgd
import lasagne.updates



def run_task(*_):
    # env = normalize(HalfCheetahEnv())

    env = normalize(GymEnv(env_name = "MountainCarContinuous-v0",force_reset=True))

    # max_path_length = env.horizon
    # baseline = GaussianMLPBaseline(env_spec=env.spec)
    baseline = ZeroBaseline(env_spec=env.spec)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers
        hidden_sizes=(64, 64)
    )
    # optimizer = FirstOrderOptimizer(update_method=lasagne.updates.adam, learning_rate=1e-1)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100,
        max_path_length=100,
        n_itr=800,
        discount=0.99,
        optimizer_args=dict(
            learning_rate=0.01,
        )
    )
    algo.train()

    # algo = VPG(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     optimizer=optimizer,
    #     n_itr = 100,
    #     batch_size = 100,
    #     max_path_length = 100,
    #     discount = 0.9,
    # )
    # algo.train()

run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    log_dir='./log/vpg_mntcar_cont',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)


# https://github.com/rll/rllab/issues/146















