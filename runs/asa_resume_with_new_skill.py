#!/usr/bin/env python

import tensorflow as tf
import joblib
import os

from sandbox.asa.algos import AdaptiveSkillAcquisition
from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import MinibotForwardPolicy, MinibotLeftPolicy
from sandbox.asa.utils.network import MLP            # MLP for new top policy

from garage.tf.algos import TRPO                     # Policy optimization algorithm
from garage.tf.baselines import GaussianMLPBaseline  # Baseline for Advantage function { A(s, a) = Q(s, a) - B(s) }
from sandbox.asa.envs import MinibotEnv              # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy  # Policy networks
from garage.misc.instrument import run_experiment    # Experiment-running util
from garage.tf.core.layers import DenseLayer



## If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


with tf.Session().as_default():
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
    # Get old policy from saved data
    old_top_policy = saved_data['policy']

    # Get W matrices from old_top_policy  # TODO delete this segment
    out_layer = old_top_policy._l_prob
    hid_layers = []
    h_l = out_layer
    while isinstance(h_l.input_layer, DenseLayer):
        h_l = h_l.input_layer
        hid_layers.append(h_l)
    hid_layers.reverse()
    hidden_w_tf_vars = [l.w for l in hid_layers]
    hidden_b_tf_vars = [l.b for l in hid_layers]
    output_w_tf_var = out_layer.w
    output_b_tf_var = out_layer.b

    # Get W matrices from old_top_policy  # TODO finish
    from garage.misc.tensor_utils import unflatten_tensors
    weight_values = unflatten_tensors(
        old_top_policy.get_param_values(),
        old_top_policy.get_param_shapes()
    )
    weights = zip(
        [p.name for p in old_top_policy.get_params()],
        weight_values
    )

    # Create new MLP using W matrices  # TODO delete? will we need custom MLP when using policy.set_param_values()?
    new_prob_network = MLP(  # Creating asa.util.network.MLP, derived from garage.tf.core.network.MLP
        # Parameters used to create original MLP (from CategoricalMLPPolicy)
        input_shape=(tf_hrl_env.spec.observation_space.flat_dim,),
        output_dim=tf_hrl_env.spec.action_space.n,
        hidden_sizes=(32, 32),              # As was in asa_test.py
        hidden_nonlinearity=tf.nn.tanh,     # Default from CategoricalMLPPolicy
        output_nonlinearity=tf.nn.softmax,  # Fixed value from CategoricalMLPPolicy
        name="prob_network",                # Fixed value from CategoricalMLPPolicy
        # Pre-trained weight matrices
        hidden_w_tf_vars=hidden_w_tf_vars,
        hidden_b_tf_vars=hidden_b_tf_vars,
        output_w_tf_var=output_w_tf_var,
        output_b_tf_var=output_b_tf_var
    )

    # DEBUG ........ this is runnable, and pickle can NOT pickle it ........
    from garage.tf.core.network import MLP as GarageMLP
    import numpy as np

    debug_output_w_1 = tf.Variable(name='dummy1', initial_value=tf.zeros((32, 2)))
    debug_output_w_2 = tf.get_variable(
        name='dummy2',
        shape=(32, 2),
        initializer=tf.zeros_initializer(),
        trainable=False,
        regularizer=None,
        dtype=tf.float32)
    init_debug_op = tf.variables_initializer([debug_output_w_1])
    init_debug_op.run()

    new_prob_network = GarageMLP(
        input_shape = (25,),
        output_dim = 2,
        hidden_sizes = (32, 32),
        hidden_nonlinearity = tf.nn.tanh,
        output_nonlinearity = tf.nn.tanh,
        # output_w_init=debug_output_w_1,
        name='dummy'
    )
    tf.global_variables_initializer().run()  # to init all variables in new_prob_network
    # exit()
    # DEBUG ........ end ........

    # Create new_policy using MLP as prob_network  # TODO Create empty policy instead
    new_top_policy = CategoricalMLPPolicy(
        env_spec=tf_hrl_env.spec,
        hidden_sizes=(32, 32),  # As was in asa_test.py
        prob_network=new_prob_network
    )

    # TODO! Fill policy with values using Parametrized.set_param_values()

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
    # new_data['policy'] = new_top_policy  # TODO! "TypeError: can't pickle _thread.RLock objects"
    # new_data['algo'] = asa_algo          # TODO! "TypeError: can't pickle _EagerContext objects"
    # new_data['can we pickle this? no'] = new_prob_network
    # new_data['can we pickle this? no'] = debug_output_w_1
    new_data['can we pickle this? no'] = debug_output_w_2
    joblib.dump(new_data, '/home/h/holas3/garage/data/local/asa-test/instant-run/itr_11_edited.pkl', compress=3)

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
