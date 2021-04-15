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
from sandbox.asa.policies import GridworldTargetPolicy, GridworldStepPolicy, GridworldRandomPolicy, GridworldStayPolicy
from sandbox.asa.policies import CategoricalMLPSkillIntegrator

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.envs import TfEnv                     # Environment wrapper
from garage.tf.policies import CategoricalMLPPolicy  # Policy networks
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors



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
                '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Basic_runs/2021_02_02-09_50--Basic_run_M2_13r4d_6coin_7step_300itrs--s4/itr_79.pkl'
                # DEBUG For direct runs: path to snapshot file (itr_N.pkl) to start from
snapshot_name = os.path.splitext(os.path.basename(snapshot_file))[0]
new_skill_policy_file = args.skill_policy or \
                '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Skill_policies/Skill_Jvvv_rpos_b20k_mpl800--good_a0.25/2021_02_17-22_01--after_itr_79--Skill_Jvvv_rpos_b20k_mpl800--s4/final.pkl'
                # DEBUG For direct runs: path to file with new skill policy

# DEBUG For runs without loaded skill - to use Gridworld*Policy as new skill
new_skill_policy_file = None
# skill_policy_exp_name = 'GWTarget'
# skill_policy_exp_name = 'GWRandom_25'
skill_policy_exp_name = 'GWStay_25'
new_skill_subpath = {
    'actions': [15, 15, 15],
    'start_observations': np.array([[21, 47,  1,  0, 0, 1, 0, 0, 1],  # in region I, holding coin, some coins picked
                                    [21, 47,  1,  0, 1, 1, 0, 1, 1],
                                    [21, 47,  1,  1, 1, 1, 1, 1, 1],
                                   ])
}
# /DEBUG



## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if int(args.seed) % 2 == 0 else '1'



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
            unique_end_obss = np.unique(new_skill_subpath['end_observations'], axis=0)
            new_skill_stop_func = lambda path: (path['observations'][-1] == unique_end_obss).all(axis=1).any()

        ## Lower level environment & policies
        # Base (original) environment.
        base_env = saved_data['env'].env.env  # <NormalizedEnv<MinibotEnv instance>>

        # Skill policies, operating in base environment
        skill_targets = [  # 13 basic room regions
            ( 6,  5), ( 6, 18), ( 6, 33), ( 6, 47), ( 6, 61),
            (21,  5), (21, 18), (21, 33), (21, 47), (21, 61),
            (37,  5), (37, 18), (37, 33),
        ]
        trained_skill_policies = \
            [GridworldTargetPolicy(env_spec=base_env.spec, target=t) for t in skill_targets] + \
            [GridworldStepPolicy(env_spec=base_env.spec, direction=d, n=7) for d in range(4)] + \
            [
             # new_skill_policy
             # GridworldTargetPolicy(env_spec=base_env.spec, target=(43, 54))  # DEBUG use GridworldTargetPolicy as new skill
             # GridworldRandomPolicy(env_spec=base_env.spec, n=25)             # DEBUG use GridworldRandomPolicy as new skill
             GridworldStayPolicy(env_spec=base_env.spec, n=25)             # DEBUG use GridworldStayPolicy as new skill
            ]
        trained_skill_policies_stop_funcs = \
                [pol.skill_stopping_func for pol in trained_skill_policies[:-1]] + \
                [
                 # new_skill_stop_func
                 trained_skill_policies[-1].skill_stopping_func                  # DEBUG use Gridworld*Policy as new skill
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
                name="TopCategoricalMLPPolicy2"
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
                skill_max_timesteps=150  # TODO? should match number in asa_basic_run
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
                    max_path_length=50,
                    n_itr=300,
                    start_itr=saved_data['itr'] + 1,  # Continue from previous iteration number
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
exp_name_extra = 'From_all'  # Name of run

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
