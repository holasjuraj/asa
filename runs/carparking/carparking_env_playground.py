import tensorflow as tf

from garage.envs.box2d import CarParkingEnv
from garage.envs import normalize
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy, ContinuousMLPPolicy
from garage.tf.baselines import GaussianMLPBaseline
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.algos import TRPO, DDPG
from garage.replay_buffer import SimpleReplayBuffer
from garage.misc.instrument import run_experiment

from datetime import datetime


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def run_task(*_):
    env = TfEnv(normalize(CarParkingEnv(
        random_start_range=0.25
    )))

    # TRPO
    policy = GaussianMLPPolicy(
        name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=20_000,
        max_path_length=400,
        n_itr=200,
        discount=0.99,
        # plot=True
    )

    # # DDPG
    # policy = ContinuousMLPPolicy(
    #     name="policy", env_spec=env.spec, hidden_sizes=(32, 32))
    #
    # replay_buffer = SimpleReplayBuffer(
    #     env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)
    #
    # qf = ContinuousMLPQFunction(
    #     env_spec=env.spec,
    #     hidden_sizes=[64, 64],
    #     hidden_nonlinearity=tf.nn.relu)
    #
    # algo = DDPG(
    #     env=env,
    #     policy=policy,
    #     qf=qf,
    #     replay_buffer=replay_buffer,
    #     exploration_strategy=OUStrategy(env.spec, sigma=0.2),
    #     n_epochs=200,
    #     n_epoch_cycles=10,
    #     max_path_length=2000,
    #     discount=0.99,
    #     # plot=True,
    #     # pause_for_plot=True
    # )

    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # Train RL agent
        algo.train(sess=session)


# Launch training
seed = 1
run_experiment(
        run_task,
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='carparking-env-playground',
        exp_name=datetime.now().strftime('%Y_%m_%d-%H_%M') + '-CarParkingEnv-r2-rng025-pathlen400',
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="none",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed,
        # Plot after each batch
        # plot=True
)
