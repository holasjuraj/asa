#!/usr/bin/env python

import tensorflow as tf
import os
import argparse
from datetime import datetime

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
from garage.misc.instrument import run_experiment    # Experiment-running util


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Parse arguments
parser = argparse.ArgumentParser(description='Resume ASA training with new skill')
parser.add_argument('-s', '--seed',
                    help='specific randomization seed, "random" for random seed, '
                         '"keep" to keep seed specified in training script. Default: "keep"',
                    metavar='SEED', default='keep')
args = parser.parse_args()


# plot = True
# if plot:
#     # Workaround to create Qt application in main thread
#     import matplotlib
#     matplotlib.use('qt5Agg')
#     import matplotlib.pyplot as plt
#     plt.figure()


def run_task(*_):

    ## Lower level environment & policies
    # Base (original) environment.
    base_env = normalize(
                MinibotEnv(
                    use_maps='all',  # [0,1]
                    discretized=True,
                    states_cache=dict()
                )
    )
    tf_base_env = TfEnv(base_env)

    # Skill policies, operating in base environment
    trained_skill_policies = [
            MinibotForwardPolicy(env_spec=base_env.spec),
            MinibotLeftPolicy(env_spec=base_env.spec)
    ]
    trained_skill_policies_stop_funcs = [
            lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
            lambda path: len(path['actions']) >= 3   # 3 steps to rotate 90°
    ]
    skill_policy_prototype = GaussianMLPPolicy(
            env_spec=tf_base_env.spec,
            hidden_sizes=(64, 64)
    )

    ## Upper level environment & policies
    # Hierarchized environment
    hrl_env = HierarchizedEnv(
            env=base_env,
            num_orig_skills=len(trained_skill_policies)
    )
    tf_hrl_env = TfEnv(hrl_env)

    # Top policy
    top_policy = CategoricalMLPPolicy(
            env_spec=tf_hrl_env.spec,
            hidden_sizes=(32, 32)
    )

    # Hierarchy of policies
    hrl_policy = HierarchicalPolicy(
            top_policy=top_policy,
            skill_policy_prototype=skill_policy_prototype,
            skill_policies=trained_skill_policies,
            skill_stop_functions=trained_skill_policies_stop_funcs,
            skill_max_timesteps=20
    )
    # Link hrl_policy and hrl_env, so that hrl_env can use skills
    hrl_env.set_hrl_policy(hrl_policy)

    ## Other
    # Baseline
    baseline = GaussianMLPBaseline(env_spec=tf_hrl_env.spec)

    # Main ASA algorithm
    asa_algo = AdaptiveSkillAcquisition(
            env=tf_hrl_env,
            hrl_policy=hrl_policy,
            baseline=baseline,
            top_algo_cls=TRPO,
            low_algo_cls=TRPO,
            # Top algo kwargs
                batch_size=5000,
                max_path_length=100,
                n_itr=80,
                discount=0.9,
                force_batch_sampler=True,
            low_algo_kwargs={
                'batch_size': 2500,
                'max_path_length': 50,
                'n_itr': 500,
                'discount': 0.99,
            }
    )

    ## Launch training
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # Train HRL agent
        asa_algo.train(sess=session)


## Run directly
# run_task()


## Run pickled
# # Erase snapshots from previous instant run
# import shutil
# shutil.rmtree('/home/h/holas3/garage/data/local/asa_test/instant_run', ignore_errors=False)

# General experiment settings
seed = 3                    # Will be ignored if --seed option is used
exp_name_direct = None      # If None, exp_name will be constructed from exp_name_extra and other info. De-bug value = 'instant_run'
exp_name_extra = 'Basic_run_80itrs_6maps_pnl005_disc09'  # Name of run

# Seed
seed = seed if args.seed == 'keep' \
       else None if args.seed == 'random' \
       else int(args.seed)

# Launch training
run_experiment(
        run_task,
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-test',
        exp_name=exp_name_direct or \
                 (datetime.now().strftime('%Y_%m_%d-%H_%M')
                  + (('--' + exp_name_extra) if exp_name_extra else '')
                  + (('--s' + str(seed)) if seed else '')
                 ),
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="all",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed,
        # Plot after each batch
        # plot=True
)
