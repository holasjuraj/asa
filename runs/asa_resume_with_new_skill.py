#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import joblib
import argparse
import os
from datetime import datetime

from sandbox.asa.algos import AdaptiveSkillAcquisition
from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import MinibotForwardPolicy, MinibotLeftPolicy, MinibotRightPolicy
from sandbox.asa.policies import CategoricalMLPSkillIntegrator

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.envs import TfEnv                     # Environment wrapper
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy  # Policy networks
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Parse arguments
parser = argparse.ArgumentParser(description='Resume ASA training with new skill')
parser.add_argument('-s', '--snapshot', help='path to snapshot file (itr_N.pkl) to start from', metavar='FILE')
args = parser.parse_args()

snapshot_file = args.snapshot or '/home/h/holas3/garage/data/local/asa-test/itr_0.pkl'  # for direct runs
snapshot_name = os.path.splitext(os.path.basename(snapshot_file))[0]


def run_task(*_):
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as tf_session:
        ## Load data from itr_N.pkl
        saved_data = joblib.load(snapshot_file)

        ## Lower level environment & policies
        # Base (original) environment.
        base_env = saved_data['env'].env.env
        tf_base_env = TfEnv(base_env)

        # Skill policies, operating in base environment
        trained_skill_policies = [
                MinibotForwardPolicy(env_spec=base_env.spec),
                MinibotLeftPolicy(env_spec=base_env.spec),
                # TODO! use trained new skill policy instead of MinibotRightPolicy
                MinibotRightPolicy(env_spec=base_env.spec)
        ]
        trained_skill_policies_stop_funcs = [
                lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
                lambda path: len(path['actions']) >= 3,  # 3 steps to rotate 90°
                # TODO! use stop-func for trained new skill policy instead of MinibotRightPolicy
                lambda path: len(path['actions']) >= 3,  # 3 steps to rotate 90°
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


        ## Top policy
        # 1) Get old policy from saved data
        old_top_policy = saved_data['policy']

        # 2) Get weights of old top policy
        otp_weights = unflatten_tensors(
                old_top_policy.get_param_values(),
                old_top_policy.get_param_shapes()
        )

        # 3) Create weights for new top policy
        skill_integrator = CategoricalMLPSkillIntegrator()
        ntp_weight_values = skill_integrator.integrate_skill(
                old_policy_weights=otp_weights,
                method=skill_integrator.Method.OLD_SKILL_AVG
        )

        # 4) Create new policy and randomly initialize its weights
        new_top_policy = CategoricalMLPPolicy(
                env_spec=tf_hrl_env.spec,  # This env counts with new skill (action space = n + 1)
                hidden_sizes=(32, 32),     # As was in asa_test.py,
                name="CategoricalMLPPolicy2"
        )
        ntp_init_op = tf.variables_initializer(new_top_policy.get_params())
        ntp_init_op.run()

        # 5) Fill new policy with adjusted weights
        new_top_policy.set_param_values(
                flattened_params=flatten_tensors(ntp_weight_values)
        )


        ## Hierarchy of policies
        hrl_policy = HierarchicalPolicy(
                top_policy=new_top_policy,
                skill_policy_prototype=skill_policy_prototype,
                skill_policies=trained_skill_policies,
                skill_stop_functions=trained_skill_policies_stop_funcs,
                skill_max_timesteps=20
        )
        # Link hrl_policy and hrl_env, so that hrl_env can use skills
        hrl_env.set_hrl_policy(hrl_policy)

        ## Other
        # Baseline
        baseline = saved_data['baseline']  # Take trained baseline

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
                    n_itr=25,
                    start_itr=saved_data['itr'] + 1,  # Continue from previous iteration number
                    discount=0.99,
                    force_batch_sampler=True,
                low_algo_kwargs={
                    'batch_size': 1000,
                    'max_path_length': 30,
                    'n_itr': 25,
                    'discount': 0.99,
                }
        )

        ## Launch training
        asa_algo.train(sess=tf_session)


## Run directly
# run_task()

## Run pickled
seed = 1
exp_name_direct = None  # 'instant_run'
exp_name_extra = 'Skill_integrator_from_left_skill'

run_experiment(
        run_task,
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-resume-with-new-skill',
        exp_name=exp_name_direct or \
                 (datetime.now().strftime('%Y_%m_%d-%H_%M')
                  + '--resumed_from_' + snapshot_name
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
