#!/usr/bin/env python

import tensorflow as tf
import joblib

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
from garage.tf.core.network import MLP               # MLP for new top policy
from garage.misc.instrument import run_experiment    # Experiment-running util


def run_task(*_):
    ## Load data from itr_N.pkl
    pkl_file = 'data/local/asa-test/instant-run/itr_11.pkl'
    with tf.Session().as_default():
        saved_data = joblib.load(pkl_file)

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

    # TODO Get W matrix from old_policy

    # TODO Create new MLP using W matrix
    new_prob_network = MLP(
            # input_shape=(env_spec.observation_space.flat_dim,),
            # output_dim=env_spec.action_space.n,
            # hidden_sizes=hidden_sizes,
            # hidden_nonlinearity=hidden_nonlinearity,
            # output_nonlinearity=tf.nn.softmax,
            # name=self._prob_network_name,
    )

    # Create new_policy using MLP as prob_network
    new_top_policy = CategoricalMLPPolicy(
            env_spec=tf_hrl_env.spec,
            hidden_sizes=(32, 32),
            prob_network=new_prob_network
    )

    # Hierarchy of policies
    hrl_policy = HierarchicalPolicy(
            top_policy=new_top_policy,
            skill_policy_prototype=skill_policy_prototype,
            skill_policies=trained_skill_policies,
            skill_stop_functions=trained_skill_policies_stop_funcs,
            skill_max_timesteps=20
    )
    # TODO << not revised after this point >>
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
                n_itr=25,
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
    # Configure TF session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # Train HRL agent
        asa_algo.train(sess=session)


## Run pickled
# # Erase snapshots from previous instant run
# import shutil
# shutil.rmtree('/home/h/holas3/garage/data/local/asa-test/instant-run', ignore_errors=False)
# Run experiment
seed = 1
run_experiment(
        run_task,
        # Resume from edited snapshot
        resume_from='data/local/asa-test/instant-run/itr_11_edited.pkl',
        # Configure TF
        use_tf=True,
        use_gpu=True,
        # Name experiment
        exp_prefix='asa-test',   # TODO? rename
        exp_name='instant-run',  # TODO? rename
        # Number of parallel workers for sampling
        n_parallel=0,
        # Snapshot information
        snapshot_mode="all",
        # Specifies the seed for the experiment  (random seed if None)
        seed=seed,
        # Plot after each batch
        # plot=True
)
