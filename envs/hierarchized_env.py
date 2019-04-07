import numpy as np

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from gym import Wrapper
from garage.envs.base import Step
from garage.envs import EnvSpec
from gym.spaces import Discrete
from sandbox.asa.sampler import skill_rollout


class HierarchizedEnv(Wrapper, Serializable):
    def __init__(
            self,
            env,
            hrl_policy
    ):
        """
        Creates a top-level environment for a HRL agent. Original env`s actions are replaced by N discrete actions,
        N being the number of skills.
        :param env: AsaEnv environment to wrap
        :param hrl_policy: A HierarchicalPolicy containing all current skill policies
        """
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self.hrl_policy = hrl_policy
        self.action_space = self.create_hrl_env_spec(self.env, self.num_skills).action_space

    @property
    def num_skills(self):
        return self.hrl_policy.num_skills

    @overrides
    def step(self, action):
        skill_path = skill_rollout(env=self.env,  # TODO? Maybe TfEnv(self.base_env) will be needed.
                                   agent=self.hrl_policy.get_skill_policy(action),
                                   skill_stopping_func=self.hrl_policy.get_skill_stopping_func(action),
                                   max_path_length=self.hrl_policy.skill_max_timesteps,
                                   reset_start_rollout=False  # do not reset the env, continue from current state
                                   )
        next_obs = self.env.get_current_obs()
        reward = np.sum(skill_path['rewards'])
        term = skill_path['terminated'][-1]
        return Step(next_obs, reward, term)

    @overrides
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    @staticmethod
    def create_hrl_env_spec(base_env, num_skills):
        return EnvSpec(
            observation_space=base_env.observation_space,
            action_space=Discrete(num_skills)
        )
