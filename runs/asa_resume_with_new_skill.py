#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import joblib
import os

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
from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors

## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config).as_default():
    ## Load data from itr_N.pkl
    pkl_file = '/home/h/holas3/garage/data/local/asa-test/itr_11.pkl'
    saved_data = joblib.load(pkl_file)
    new_data = dict(saved_data)  # New data to be dumped once edited

    ## Lower level environment & policies
    # Base (original) environment.
    base_env = saved_data['env'].env.env
    tf_base_env = TfEnv(base_env)

    # Skill policies, operating in base environment
    trained_skill_policies = [
        MinibotForwardPolicy(env_spec=base_env.spec),
        MinibotLeftPolicy(env_spec=base_env.spec),
        # TODO! new skill policy
    ]
    trained_skill_policies_stop_funcs = [
        lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
        lambda path: len(path['actions']) >= 3,  # 3 steps to rotate 90°
        # TODO! new skill stopping function
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
    otp_weight_values = unflatten_tensors(
        old_top_policy.get_param_values(),
        old_top_policy.get_param_shapes()
    )
    otp_weights = zip(
        [p.name for p in old_top_policy.get_params()],
        otp_weight_values
    )

    # 3) Create weights for new top policy
    ntp_weights = [(name, np.copy(value)) for name, value in otp_weights]
    # TODO! adjust weights to integrate new skill
    ntp_weight_values = [value for _, value in ntp_weights]

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

    ## Save edited data
    new_data['env'] = tf_hrl_env
    new_data['policy'] = new_top_policy
    new_data['algo'] = asa_algo          # TODO! "_pickle.PicklingError: Can't pickle <class 'garage.tf.misc.tensor_utils.PolicyOptInputs'>: it's not found as garage.tf.misc.tensor_utils.PolicyOptInputs"

    # DEBUG
    from garage.tf.algos import NPO
    new_data['can we pickle this? no'] = NPO(policy=new_top_policy,
                                             env=tf_hrl_env,
                                             baseline=baseline)


    # "TypeError: can't pickle _EagerContext objects"
    #
    # Reason:
    # AdaptiveSkillAcquisition uses TRPO, TRPO inherits NPO, NPO uses tf.name_scope,
    # name_scope uses tf.python.context.context(), context uses _EagerContext,
    # _EagerContext inherits threading.local which cannot be pickled
    #
    # Possible solutions:
    # A) Implement __getstate__() and __setstate__() in NPO class, that will
    # export/import everything *except* self._name_scope .
    # B) In NPO, change self._name_scope to local variable _name_scope, and
    # hope that it won`t mess up other things.
    #
    # Chosen solution: B, edit garage.tf.algos.npo.NPO.__init__

    joblib.dump(new_data, '/home/h/holas3/garage/data/local/asa-test/itr_11_edited.pkl', compress=3)

print('################ SNAPSHOT FILE EDITING COMPLETE ################')
exit()  # DEBUG




## Resume training
seed = 1
run_experiment(
        # Method call must be provided, but it will not be called if we use resume_from
        method_call=lambda: None,
        # Resume from edited snapshot
        resume_from='/home/h/holas3/garage/data/local/asa-test/instant-run/itr_11_edited.pkl',
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-test',
        exp_name='instant-run-resumed',
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="all",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed,
        # Plot after each batch
        # plot=True
)
