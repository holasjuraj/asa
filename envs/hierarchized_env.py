import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces import Discrete
from rllab.misc.overrides import overrides
from sandbox.asa.sampler.utils import rollout


class HierarchizedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            hrl_policy
    ):
        """
        Creates a top-level environment for a HRL agent. Original env`s actions are replaced by N discrete actions,
        N being the number of skills.
        :param env: Environment to wrap
        :param hrl_policy: A HierarchicalPolicy containing all current skill policies
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.hrl_policy = hrl_policy

    @property
    def num_skills(self):
        return self.hrl_policy.num_skills

    @overrides
    @property
    def action_space(self):
        return Discrete(self.num_skills)

    @overrides
    def step(self, action):
        skill_path = rollout(env=self._wrapped_env,
                             agent=self.hrl_policy.get_skill_policy(action),
                             max_path_length=self.hrl_policy.skill_max_timesteps,
                             reset_start_rollout=True  # do not reset the env, continue from current state
                             )
        # TODO wrapped environment must support get_current_obs()
        next_obs = self.wrapped_env.get_current_obs()
        reward = np.sum(skill_path['rewards'])
        term = skill_path['terminated'][-1]
        return Step(next_obs, reward, term)
