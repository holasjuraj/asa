#!/usr/bin/env python

import tensorflow as tf

from sandbox.asa.algos import AdaptiveSkillAcquisition
from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import MinibotForwardPolicy, MinibotLeftPolicy

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.baselines import GaussianMLPBaseline  # Baseline for Advantage function { A(s, a) = Q(s, a) - B(s) }
from sandbox.asa.envs import MinibotEnv              # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy  # Policy networks
# from garage.misc.instrument import run_experiment    # Experiment-running util


plot = True
# if plot:
#     # Workaround to create Qt application in main thread
#     import matplotlib
#     matplotlib.use('qt5Agg')
#     import matplotlib.pyplot as plt
#     plt.figure()


def run_task(*_):

    # Base (original) environment
    base_env = TfEnv(
                normalize(
                    MinibotEnv(
                        use_maps=[0, 1],  # 'all',  # [0,1]
                        discretized=True
    )))

    # Skill policies
    trained_skill_policies = [
        MinibotForwardPolicy(env_spec=base_env.spec),
        MinibotLeftPolicy(env_spec=base_env.spec)
    ]
    trained_skill_policies_stop_funcs = [
        lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
        lambda path: len(path['actions']) >= 3   # 3 steps to rotate 90°
    ]
    skill_policy_prototype = GaussianMLPPolicy(
        env_spec=base_env.spec,
        hidden_sizes=(64, 64)
    )
    # Top policy
    top_policy = CategoricalMLPPolicy(
        env_spec=HierarchizedEnv.create_hrl_env_spec(
            base_env=base_env,
            num_skills=len(trained_skill_policies)
        ),
        hidden_sizes=(32, 32)
    )

    # Hierarchy of policies
    hrl_policy = HierarchicalPolicy(
        env_spec=base_env.spec,
        top_policy=top_policy,
        skill_policy_prototype=skill_policy_prototype,
        skill_policies=trained_skill_policies,
        skill_stop_functions=trained_skill_policies_stop_funcs,
        skill_max_timesteps=20
    )

    # Hierarchized environment
    hrl_env = HierarchizedEnv(
            env=base_env,
            hrl_policy=hrl_policy
    )

    # Baseline
    baseline = GaussianMLPBaseline(env_spec=hrl_env.spec)

    # Main ASA algorithm
    asa_algo = AdaptiveSkillAcquisition(
            env=hrl_env,
            hrl_policy=hrl_policy,
            baseline=baseline,
            top_algo_cls=TRPO,
            low_algo_cls=TRPO,
            # Top algo kwargs
                batch_size=1000,
                max_path_length=100,
                n_itr=15,
                discount=0.99,
            low_algo_kwargs={
                'batch_size': 1000,
                'max_path_length': 30,
                'n_itr': 25,
                'discount': 0.99,
            }
    )

    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # Train HRL agent
        asa_algo.train(sess=session)


# Run directly
run_task()

# # Run pickled
# for seed in range(1, 6):
#     run_experiment(
#         run_task,
#         exp_prefix='asa-test',
#         exp_name='{}'.format(seed),
#         # Number of parallel workers for sampling
#         n_parallel=2,
#         # Only keep the snapshot parameters for the last iteration
#         snapshot_mode="last",
#         # Specifies the seed for the experiment. If this is not provided, a random seed will be used
#         seed=seed,
#     #     plot=True
#     )
