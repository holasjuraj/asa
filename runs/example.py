#!/usr/bin/env python

# Policy optimization algorithm
from builtins import input

from rllab.algos.trpo import TRPO

# Baseline for Advantage function { A(s) = V(s) - B(s) }
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

# Environment
from sandbox.asa.envs.grid_maze_env import GridMazeEnv
from rllab.envs.normalized_env import normalize

# Policy network
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

# Experiment-running util
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    env = normalize(GridMazeEnv(plot={
                                    'visitation': {#'save':'~/rllab/data/local/asl-example/instant-run',
                                                   # 'save':'~/rllab/data/local/asl-example/aggregation-plot-2018_09_26_16_30',
                                                   # 'live': True
                                                   },
                                    'aggregation': {#'save':'~/rllab/data/local/asl-example/instant-run',
                                                   # 'save':'~/rllab/data/local/asl-example/aggregation-plot-2018_09_26_16_30',
                                                   # 'live': True
                                                   }
                                },
                                use_maps='all',  # [0,1]
                                ))

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=100,
        n_itr=25,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True
    )
    algo.train()


# Run directly
run_task()
input('< Press Enter to quit >')  # Prevent plots from closing

# # Run pickled
# for seed in range(1, 6):
#     run_experiment_lite(
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
