#!/usr/bin/env python

import tensorflow as tf
import dill
import argparse
import os
from datetime import datetime
import numpy as np

from sandbox.asa.algos import AdaptiveSkillAcquisition
from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import MinibotForwardPolicy, MinibotLeftPolicy, MinibotRightPolicy, MinibotRandomPolicy
from sandbox.asa.policies import CategoricalMLPSkillIntegrator

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.envs import TfEnv                     # Environment wrapper
from garage.tf.policies import CategoricalMLPPolicy  # Policy networks
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors


## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Parse arguments
parser = argparse.ArgumentParser(description='Resume ASA training with new skill')
parser.add_argument('-f', '--file',
                    help='path to snapshot file (itr_N.pkl) to start from', metavar='FILE')
parser.add_argument('-p', '--skill-policy',
                    help='path to file with new skill policy', metavar='FILE')
parser.add_argument('-i', '--integration-method',
                    help='integration method to be used, number (index of method) or '
                         '"keep" to keep method specified in training script. Default: "keep"',
                    metavar='METHOD', default='keep')
parser.add_argument('-s', '--seed',
                    help='specific randomization seed, "random" for random seed, '
                         '"keep" to keep seed specified in training script. Default: "keep"',
                    metavar='SEED', default='keep')
args = parser.parse_args()

snapshot_file = args.file or \
                '/home/h/holas3/garage/data/archive/TEST4_Resumed_80itrs_discount0.99/Basic_runs/2020_03_09-21_34--Basic_run_80itrs_6maps_disc099_b5000--s1/itr_3.pkl'
                # For direct runs: path to snapshot file (itr_N.pkl) to start from
snapshot_name = os.path.splitext(os.path.basename(snapshot_file))[0]
new_skill_policy_file = args.skill_policy or \
                '/home/h/holas3/garage/sandbox/asa/data/local/asa-train-new-skill/2020_03_10-15_10--after_itr_3--For_all_disc099_Skill_sLLLs--s1/final.pkl'
                # For direct runs: path to file with new skill policy

# # DEBUG For runs without loaded skill - to use Minibot*Policy as new skill
# new_skill_policy_file = None
# # skill_policy_exp_name = 'MinibotRight'
# skill_policy_exp_name = 'MinibotRandom'
# new_skill_subpath = {
#     'actions': [1, 1, 1],
#     'start_observations': np.array([[0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  # at corner
# }



def run_task(*_):
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config).as_default() as tf_session:
        ## Load data from itr_N.pkl
        with open(snapshot_file, 'rb') as file:
            saved_data = dill.load(file)

        ## Load data of new skill
        global new_skill_subpath
        if new_skill_policy_file:
            with open(new_skill_policy_file, 'rb') as file:
                new_skill_data = dill.load(file)
            new_skill_policy = new_skill_data['policy']
            new_skill_subpath = new_skill_data['subpath']
            new_skill_stop_func = lambda path: path['observations'][-1] in new_skill_subpath['end_observations']

        ## Lower level environment & policies
        # Base (original) environment.
        base_env = saved_data['env'].env.env  # <NormalizedEnv<MinibotEnv instance>>

        # Skill policies, operating in base environment
        trained_skill_policies = [
                MinibotForwardPolicy(env_spec=base_env.spec),
                MinibotLeftPolicy(env_spec=base_env.spec),
                new_skill_policy
                # MinibotRightPolicy(env_spec=base_env.spec)   # DEBUG use MinibotRightPolicy as new skill
                # MinibotRandomPolicy(env_spec=base_env.spec)  # DEBUG use MinibotRandomPolicy as new skill
        ]
        trained_skill_policies_stop_funcs = [
                lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
                lambda path: len(path['actions']) >= 3,  # 3 steps to rotate 90°
                new_skill_stop_func
                # lambda path: len(path['actions']) >= 3  # 3 steps to rotate 90°     # DEBUG use MinibotRightPolicy as new skill
                # lambda path: len(path['actions']) >= 5  # 5 steps for random policy # DEBUG use MinibotRandomPolicy as new skill
        ]
        skill_policy_prototype = saved_data['hrl_policy'].skill_policy_prototype

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
                method=skill_integration_method,
                # Specific parameters for START_OBSS_SKILLS_AVG
                subpath_start_obss=new_skill_subpath['start_observations'],
                top_policy=old_top_policy,
                # Specific parameters for SUBPATH_SKILLS_AVG, SUBPATH_SKILLS_SMOOTH_AVG and SUBPATH_FIRST_SKILL
                subpath_actions=new_skill_subpath['actions']
        )

        # 4) Create new policy and randomly initialize its weights
        new_top_policy = CategoricalMLPPolicy(
                env_spec=tf_hrl_env.spec,  # This env counts with new skill (action space = n + 1)
                hidden_sizes=(32, 32),     # As was in asa_basic_run.py,
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
                skill_max_timesteps=50
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
                    n_itr=80,
                    start_itr=saved_data['itr'] + 1,  # Continue from previous iteration number
                    discount=0.9,
                    force_batch_sampler=True,
                low_algo_kwargs={
                    'batch_size': 1000,
                    'max_path_length': 30,
                    'n_itr': 25,
                    'discount': 0.99,
                }
        )

        ## Launch training
        train_info = asa_algo.train(
                sess=tf_session,
                snapshot_mode='none'
        )

        ## Save last iteration
        out_file = os.path.join(train_info['snapshot_dir'], 'final.pkl')
        empty_samples_data = {'paths': None}
        with open(out_file, 'wb') as file:
            out_data = asa_algo.get_itr_snapshot(
                itr=asa_algo.n_itr - 1,
                samples_data=empty_samples_data
            )
            dill.dump(out_data, file)


## Run directly
# run_task()


## Run pickled
# General experiment settings
seed = 3                    # Will be ignored if --seed option is used
exp_name_direct = None      # If None, exp_name will be constructed from exp_name_extra and other info. De-bug value = 'instant_run'
exp_name_extra = 'From_all_manual'  # Name of run

# Skill policy experiment name
if new_skill_policy_file:
    skill_policy_dir = os.path.basename(os.path.dirname(new_skill_policy_file))
    try: skill_policy_exp_name = skill_policy_dir.split('--')[-2]
    except IndexError: skill_policy_exp_name = skill_policy_dir

# Skill integration method
skill_integration_method = CategoricalMLPSkillIntegrator.Method.SUBPATH_SKILLS_AVG
skill_integration_method = \
        skill_integration_method.value if args.integration_method == 'keep' \
        else CategoricalMLPSkillIntegrator.get_method_str_by_index(int(args.integration_method))
skill_integration_idx = CategoricalMLPSkillIntegrator.get_index_of_method_str(skill_integration_method)

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
        exp_prefix='asa-resume-with-new-skill',
        exp_name=exp_name_direct or \
                 (datetime.now().strftime('%Y_%m_%d-%H_%M')
                  + '--resumed_' + snapshot_name
                  + (('--' + exp_name_extra) if exp_name_extra else '')
                  + '--skill_' + skill_policy_exp_name
                  + '--integ' + str(skill_integration_idx) + '_' + skill_integration_method
                  + (('--s' + str(seed)) if seed else '')
                 ),
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="all",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed
)
