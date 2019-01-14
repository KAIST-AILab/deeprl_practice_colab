from rllab.envs.gym_env import GymEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv

from rllab.algos.vpg import VPG
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer #sgd
import lasagne.updates


def run_task(*_):
    # env = normalize(HalfCheetahEnv())

    env = normalize(GymEnv(env_name = "Acrobot-v1",force_reset=True, record_video=True))

    max_path_length = env.horizon
    print(max_path_length)
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers
        hidden_sizes=(64, 64)
    )
    # optimizer = FirstOrderOptimizer(update_method=lasagne.updates.adam, learning_rate=1e-1)

    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=800,
        max_path_length=500,
        n_itr=10000,
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
    log_dir='./log/vpg_acrobot_baseline',
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)


# https://github.com/rll/rllab/issues/146















