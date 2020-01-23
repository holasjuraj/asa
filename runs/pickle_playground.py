import dill

import os
import tensorflow as tf
import joblib

from sandbox.asa.envs import HierarchizedEnv
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.policies import MinibotForwardPolicy, MinibotLeftPolicy

from sandbox.asa.envs import MinibotEnv              # Environment
from garage.envs import normalize                    #
from garage.tf.envs import TfEnv                     #
from garage.tf.policies import CategoricalMLPPolicy, GaussianMLPPolicy  # Policy networks

# Snippet for starting interactive TF session and reading exported training snapshot

# If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

path = '/home/h/holas3/garage/data/local/asa-test/instant_run/itr_0.pkl'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def get_funcs():
    def fStep(path): return len(path['actions']) >= 5  # 5 steps to move 1 tile
    def fLeft(path): return len(path['actions']) >= 3  # 3 steps to rotate 90°
    return [fStep, fLeft]

with tf.Session(config=config).as_default() as sess:
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

    def fStep(path): return len(path['actions']) >= 5  # 5 steps to move 1 tile
    def fLeft(path): return len(path['actions']) >= 3  # 3 steps to rotate 90°
    trained_skill_policies_stop_funcs = get_funcs()
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

    init = tf.global_variables_initializer()
    init.run()

    joblib.dump(value=hrl_policy, filename='/home/h/holas3/garage/data/local/asa-test/instant_run/tmp.pkl', compress=3)

    print('Done.')
