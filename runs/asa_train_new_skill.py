#!/usr/bin/env python

import tensorflow as tf
import dill
import argparse
import os
from datetime import datetime
import numpy as np


from sandbox.asa.envs import HierarchizedEnv, SkillLearningEnv
from sandbox.asa.utils.path_trie import PathTrie

from garage.tf.envs import TfEnv                     # Environment wrapper
from garage.core import Serializable
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.misc import logger


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Parse arguments
parser = argparse.ArgumentParser(description='Train new skill to be used in resumed ASA training')
parser.add_argument('-f', '--file',
                    help='path to snapshot file (itr_N.pkl) from which to train new skill', metavar='FILE')
parser.add_argument('-s', '--seed',
                    help='specific randomization seed, "random" for random seed, '
                         '"keep" to keep seed specified in training script. Default: "keep"',
                    metavar='SEED', default='keep')
args = parser.parse_args()

snapshot_file = args.file or '/home/h/holas3/garage/data/local/asa-test/2020_01_30-14_21--Basic_run_25itrs_subpth3to5_b5000--s3/itr_8.pkl'  # for direct runs
snapshot_name = os.path.splitext(os.path.basename(snapshot_file))[0]


def run_task(*_):
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as tf_session:
        ## Load data from itr_N.pkl
        with open(snapshot_file, 'rb') as file:
            saved_data = dill.load(file)


        ## Construct PathTrie and find missing skill description
        # This is basically ASA.decide_new_skill
        min_length = 3
        max_length = 5
        action_map = {0: 's', 1: 'L', 2: 'R'}
        min_f_score = 1
        max_results = 10
        aggregations = ['mean']  # sublist of ['mean', 'most_freq', 'nearest_mean', 'medoid'] or 'all'

        paths = saved_data['paths']
        path_trie = PathTrie(saved_data['hrl_policy'].num_skills)
        for path in paths:
            actions = path['actions'].argmax(axis=1).tolist()
            observations = path['observations']
            path_trie.add_all_subpaths(
                    actions,
                    observations,
                    min_length=min_length,
                    max_length=max_length
            )
        logger.log('Searched {} rollouts'.format(len(paths)))

        frequent_paths = path_trie.items(
                action_map=action_map,
                min_count=10,  # len(paths) * 2
                min_f_score=min_f_score,
                max_results=max_results,
                aggregations=aggregations
        )
        logger.log('Found {} frequent paths: [index, actions, count, f-score]'.format(len(frequent_paths)))
        for i, f_path in enumerate(frequent_paths):
            logger.log('    {:2}: {:{pad}}\t{}\t{:.3f}'.format(
                i,
                f_path['actions_text'],
                f_path['count'],
                f_path['f_score'],
                pad=max_length))

        top_subpath = frequent_paths[0]
        start_obss = top_subpath['start_observations']
        end_obss   = top_subpath['end_observations']



        ## Prepare elements for training
        # Environment
        base_env = saved_data['env'].env.env  # <NormalizedEnv<MinibotEnv instance>>
        skill_learning_env = TfEnv(
                SkillLearningEnv(
                    # base env that was wrapped in HierarchizedEnv (not fully unwrapped - may be normalized!)
                    env=base_env,
                    start_obss=start_obss,
                    end_obss=end_obss
                )
        )

        # Skill policy
        hrl_policy = saved_data['hrl_policy']
        new_skill_policy, new_skill_id = hrl_policy.create_new_skill(
                end_obss=end_obss
        )

        # Baseline - clone baseline specified in low_algo_kwargs, or top-algo`s baseline
        low_algo_kwargs = dict(saved_data['low_algo_kwargs'])
        baseline_to_clone = low_algo_kwargs.get('baseline', saved_data['baseline'])
        baseline = Serializable.clone(  # to create blank baseline
                obj=baseline_to_clone,
                name='{}Skill{}'.format(type(baseline_to_clone).__name__, new_skill_id)
        )
        low_algo_kwargs['baseline'] = baseline
        low_algo_cls = saved_data['low_algo_cls']

        # DEBUG set custom training params (should`ve been set in asa_test)
        low_algo_kwargs['batch_size'] = 2500
        low_algo_kwargs['max_path_length'] = 50
        low_algo_kwargs['n_itr'] = 50

        # Algorithm
        algo = low_algo_cls(
            env=skill_learning_env,
            policy=new_skill_policy,
            **low_algo_kwargs
        )

        # Logger parameters
        logger_snapshot_dir_before = logger.get_snapshot_dir()
        logger_snapshot_mode_before = logger.get_snapshot_mode()
        logger_snapshot_gap_before = logger.get_snapshot_gap()
        # No need to change snapshot dir in this script, it is used in ASA-algo.make_new_skill()
        # logger.set_snapshot_dir(os.path.join(
        #         logger_snapshot_dir_before,
        #         'skill{}'.format(new_skill_id)
        # ))
        logger.set_snapshot_mode('gap')
        logger.set_snapshot_gap(max(1, np.floor(low_algo_kwargs['n_itr'] / 10)))
        logger.set_tensorboard_step_key('Iteration')


        ## Train new skill
        with logger.prefix('Skill {} | '.format(new_skill_id)):
            algo.train(sess=tf_session)



        ## Save new policy and its end_obss (we`ll construct skill stopping function
        #  from them manually in asa_resume_with_new_skill.py)
        out_file = os.path.join(logger.get_snapshot_dir(), 'final.pkl')
        with open(out_file, 'wb') as file:
            out_data = {
                    'policy': new_skill_policy,
                    'end_obss': hrl_policy._skills_end_obss[new_skill_id]
            }
            dill.dump(out_data, file)

        # Restore logger parameters
        logger.set_snapshot_dir(logger_snapshot_dir_before)
        logger.set_snapshot_mode(logger_snapshot_mode_before)
        logger.set_snapshot_gap(logger_snapshot_gap_before)


## Run directly
# run_task()

## Run pickled
seed = 3
exp_name_direct = 'instant_run'
exp_name_extra = 'Skill_from_s3_itr8_path0_500itrs_len50_VectorisedSampler_Cache'

seed = seed if args.seed == 'keep' \
       else None if args.seed == 'random' \
       else int(args.seed)

run_experiment(
        run_task,
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-train-new-skill',
        exp_name=exp_name_direct or \
                 (datetime.now().strftime('%Y_%m_%d-%H_%M')
                  # + '--after_' + snapshot_name
                  + (('--' + exp_name_extra) if exp_name_extra else '')
                  + (('--s' + str(seed)) if seed else '')
                 ),
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="all",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed,
)
