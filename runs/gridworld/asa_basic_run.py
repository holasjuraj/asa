#!/usr/bin/env python

import tensorflow as tf
import os
import dill
import argparse
from datetime import datetime

from sandbox.asa.algos import AdaptiveSkillAcquisition
from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import GridworldTargetPolicy, GridworldStepPolicy

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.baselines import GaussianMLPBaseline  # Baseline for Advantage function { A(s, a) = Q(s, a) - B(s) }
from sandbox.asa.envs import GridworldGathererEnv    # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy  # Policy networks
from garage.misc.instrument import run_experiment    # Experiment-running util



# Parse arguments
parser = argparse.ArgumentParser(description='Resume ASA training with new skill')
parser.add_argument('-s', '--seed',
                    help='specific randomization seed, "random" for random seed, '
                         '"keep" to keep seed specified in training script. Default: "keep"',
                    metavar='SEED', default='keep')
args = parser.parse_args()

basic_skills_dir = '/home/h/holas3/garage/data/archive/TEST21_Resumed_all_Basic_skills/Basic_skills'

## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' if int(args.seed) % 2 == 0 else '1'



def run_task(*_):
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as tf_session:

        ## Lower level environment & policies
        # Base (original) environment.
        base_env = normalize(
                GridworldGathererEnv(
                    plot={
                        'visitation': {
                            'save': True,
                            'live': False
                        }
                    }
                )
        )
        tf_base_env = TfEnv(base_env)

        # Skill policies, operating in base environment
        skill_targets = [
            # 13 basic room regions + target
            ( 6,  5), ( 6, 18), ( 6, 33), ( 6, 47), ( 6, 61),
            (21,  5), (21, 18), (21, 33), (21, 47), (21, 61),
            (37,  5), (37, 18), (37, 33),
            (43, 54)
        ]
        trained_skill_policies = [None] * 13
        trained_skill_policies_stop_funcs = [None] * 13
        for skill_dir in os.listdir(basic_skills_dir):
            skill_id = int(skill_dir[skill_dir.find('--trg') + 5:])
            if skill_id >= 13: continue  # do not include target skill
            skill_file = os.path.join(basic_skills_dir, skill_dir, 'final.pkl')
            with open(skill_file, 'rb') as file:
                saved_data = dill.load(file)
            trained_skill_policies[skill_id] = saved_data['policy']
            trained_skill_policies_stop_funcs[skill_id] = \
                GridworldTargetPolicy(env_spec=base_env.spec, target=skill_targets[skill_id])\
                .skill_stopping_func

        skill_policy_prototype = CategoricalMLPPolicy(
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
                name="TopCategoricalMLPPolicy",
                env_spec=tf_hrl_env.spec,
                hidden_sizes=(32, 32)
        )

        # Hierarchy of policies
        hrl_policy = HierarchicalPolicy(
                top_policy=top_policy,
                skill_policy_prototype=skill_policy_prototype,
                skill_policies=trained_skill_policies,
                skill_stop_functions=trained_skill_policies_stop_funcs,
                skill_max_timesteps=150  # maximum distance in map is 108
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
                    max_path_length=50,  # ideal path is 40
                    n_itr=300,
                    discount=0.99,
                    force_batch_sampler=True,
                low_algo_kwargs={
                    'batch_size': 20000,
                    'max_path_length': 800,
                    'n_itr': 300,
                    'discount': 0.99,
                }
        )

        ## Launch training
        asa_algo.train(sess=tf_session)

        # ## Launch training
        # # Configure TF session
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(config=config) as session:
        #     # Train HRL agent
        #     asa_algo.train(sess=session)


## Run directly
# run_task()
# input('< Press Enter to quit >')  # Prevent plots from closing
# exit()


## Run pickled
# General experiment settings
seed = 3                    # Will be ignored if --seed option is used
exp_name_direct = None      # If None, exp_name will be constructed from exp_name_extra and other info. De-bug value = 'instant_run'
exp_name_extra = 'Basic_run'  # Name of run

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
        exp_prefix='asa-basic-run',
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
