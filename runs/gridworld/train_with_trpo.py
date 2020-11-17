#!/usr/bin/env python

import tensorflow as tf

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.baselines import GaussianMLPBaseline  # Baseline for Advantage function { A(s) = V(s) - B(s) }
from sandbox.asa.envs import GridWorldEnv            # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy  # Policy network
from garage.misc.instrument import run_experiment    # Experiment-running util


plot = True
# if plot:
#     # Workaround to create Qt application in main thread
#     import matplotlib
#     matplotlib.use('qt5Agg')
#     import matplotlib.pyplot as plt
#     plt.figure()


def run_task(*_):

    env = TfEnv(normalize(GridWorldEnv(
        desc='8x8_move_goal'
    )))

    policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64, 64)
    )

    baseline = GaussianMLPBaseline(env_spec=env.spec)

    algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=5000,
            max_path_length=100,
            n_itr=50,
            discount=0.99,
            step_size=0.01,
            # plot=plot,
            # pause_for_plot=False
            )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        algo.train(sess=session)


# Run directly
run_task()

# # Run pickled
# for seed in range(1, 6):
#     run_experiment(
#         run_task,
#         exp_prefix='asl-path-trie-count',
#         exp_name='all-5000-50-null-{}'.format(seed),
#         # Number of parallel workers for sampling
#         n_parallel=2,
#         # Only keep the snapshot parameters for the last iteration
#         snapshot_mode="last",
#         # Specifies the seed for the experiment. If this is not provided, a random seed will be used
#         seed=seed,
#     #     plot=True
#     )
