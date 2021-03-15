#!/usr/bin/env python

import tensorflow as tf
import dill
import argparse
import os
import numpy as np
from datetime import datetime

from sandbox.asa.envs import GridworldTargetEnv      # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy  # Policy networks
from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.baselines import GaussianMLPBaseline  # Baseline for Advantage function { A(s, a) = Q(s, a) - B(s) }
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.misc import logger



# Parse arguments
parser = argparse.ArgumentParser(description='Train basic skills to be used in all ASA trainings')
parser.add_argument('-i', '--skillid',
                    help='ID of new skill (number)', metavar='SKILLID')
parser.add_argument('-s', '--seed',
                    help='specific randomization seed, "random" for random seed, '
                         '"keep" to keep seed specified in training script. Default: "keep"',
                    metavar='SEED', default='keep')
args = parser.parse_args()

skill_id = int(args.skillid) if args.skillid is not None else 7   # DEBUG For direct runs


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' if int(args.skillid) % 2 == 0 else '1'


def run_task(*_):
    skill_targets = [
        # 13 basic room regions + target
        ( 6,  5), ( 6, 18), ( 6, 33), ( 6, 47), ( 6, 61),
        (21,  5), (21, 18), (21, 33), (21, 47), (21, 61),
        (37,  5), (37, 18), (37, 33),
        (43, 54)
    ]

    ## Prepare elements for training
    # Environment
    base_env = normalize(
            GridworldTargetEnv(
                target=skill_targets[skill_id],
                plot={
                    'visitation': {
                        'save': True,
                        'live': False
                    }
                }
            )
    )
    tf_base_env = TfEnv(base_env)

    # Policy
    policy = CategoricalMLPPolicy(
        env_spec=tf_base_env.spec,
        hidden_sizes=(64, 64),
        name='CategoricalMLPPolicySkill{}'.format(skill_id)
    )

    # Baseline
    baseline = GaussianMLPBaseline(env_spec=tf_base_env.spec)

    # Algorithm
    algo = TRPO(
        env=tf_base_env,
        policy=policy,
        baseline=baseline,
        batch_size=20000,
        max_path_length=800,
        n_itr=300,
        discount=0.99
    )

    # Logger parameters
    logger.set_tensorboard_step_key('Iteration')

    ## Launch training
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # Train RL agent
        algo.train(sess=session)

    ## Save new policy
    out_file = os.path.join(logger.get_snapshot_dir(), 'final.pkl')
    with open(out_file, 'wb') as file:
        out_data = {
                'policy': policy
        }
        dill.dump(out_data, file)


## Run directly
# run_task()


## Run pickled
# General experiment settings
seed = None                 # Will be ignored if --seed option is used
exp_name_direct = None      # If None, exp_name will be constructed from exp_name_extra and other info. De-bug value = 'instant_run'
exp_name_extra = 'Skill_Basic'  # Name of run

# Seed
seed = seed if args.seed == 'keep' \
       else None if args.seed == 'random' \
       else int(args.seed)
seed_num = np.random.randint(100) if seed is None else seed

# Launch training
run_experiment(
        run_task,
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-train-basic-skill',
        exp_name=exp_name_direct or \
                 (datetime.now().strftime('%Y_%m_%d-%H_%M')
                  + (('--' + exp_name_extra) if exp_name_extra else '')
                  + ('--trg' + str(skill_id))
                  + (('--s' + str(seed)) if seed else '')
                 ),
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="none",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed_num,
)
