import numpy as np

from sandbox.asa.envs import MinibotEnv, HierarchizedEnv
from sandbox.asa.policies import MinibotForwardPolicy, MinibotRightPolicy, MinibotLeftPolicy
from sandbox.asa.policies import HierarchicalPolicy
from sandbox.asa.sampler import skill_rollout

from garage.misc import logger
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy, CategoricalMLPPolicy



base_env = MinibotEnv(use_maps=[1])
step = MinibotForwardPolicy(base_env.spec)
left = MinibotLeftPolicy(base_env.spec)
right = MinibotRightPolicy(base_env.spec)
logger.set_snapshot_dir('/home/h/holas3/garage/data/local/minibot-env-playground')


# Skill policies, operating in base environment
tf_base_env = TfEnv(base_env)
trained_skill_policies = [
    MinibotForwardPolicy(env_spec=base_env.spec),
    MinibotLeftPolicy(env_spec=base_env.spec)
]
trained_skill_policies_stop_funcs = [
    lambda path: len(path['actions']) >= 5,  # 5 steps to move 1 tile
    lambda path: len(path['actions']) >= 3  # 3 steps to rotate 90°
]
skill_policy_prototype = GaussianMLPPolicy(
    env_spec=tf_base_env.spec,
    hidden_sizes=(64, 64)
)
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






base_env.reset()
base_env.render()

# for _ in range(15):
#     env.step(step.get_action(None)[0])
#     env.render()
#
# for _ in range(3):
#     env.step(right.get_action(None)[0])
#     env.render()
#
# for _ in range(15):
#     env.step(step.get_action(None)[0])
#     env.render()


def act(action):
    skill_rollout( env=TfEnv(base_env),
                   agent=hrl_policy.get_skill_policy(action),
                   skill_stopping_func=hrl_policy.get_skill_stopping_func(action),
                   max_path_length=hrl_policy.skill_max_timesteps,
                   reset_start_rollout=False  # do not reset the env, continue from current state
                   )
    base_env.render()


# act(0)
# print(base_env.agent_ori *180/np.pi)
# act(1)
# print(base_env.agent_ori *180/np.pi)
# act(0)
# print(base_env.agent_ori *180/np.pi)
# act(1)
# print(base_env.agent_ori *180/np.pi)
# act(0)
# print(base_env.agent_ori *180/np.pi)





for _ in range(23):
    base_env.step(step.get_action(None)[0])
    base_env.step(left.get_action(None)[0])
    base_env.render()


base_env.save_rendered_plot()
