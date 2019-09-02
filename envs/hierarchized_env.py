import numpy as np

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from gym import Wrapper
from garage.envs.base import Step
from garage.tf.envs import TfEnv
from gym.spaces import Discrete
from sandbox.asa.sampler import skill_rollout
from sandbox.asa.envs import AsaEnv


class HierarchizedEnv(Wrapper, Serializable):
    def __init__(
            self,
            env,
            num_orig_skills
    ):
        """
        Creates a top-level environment for a HRL agent. Original env`s actions are replaced by N discrete actions,
        N being the number of skills.
        :param env: AsaEnv environment to wrap
        :param num_orig_skills: number of pre-trained skill that will prepared be in HRL policy
        :param hrl_policy: A HierarchicalPolicy containing all current skill policies
        """
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self._num_orig_skills = num_orig_skills
        self.action_space = Discrete(self._num_orig_skills)
        # TODO! action_space must change after adding new skill
        self.hrl_policy = None


    def set_hrl_policy(self, hrl_policy):
        """
        :param hrl_policy: A HierarchicalPolicy containing all current skill policies
        """
        assert(self._num_orig_skills == hrl_policy.num_skills)
        self.hrl_policy = hrl_policy


    @property
    def num_skills(self):
        return self._num_orig_skills if self.hrl_policy is None else self.hrl_policy.num_skills


    @overrides
    def step(self, action):
        assert(self.hrl_policy is not None)
        skill_path = skill_rollout(env=TfEnv(self.env),  # TODO? Accept TfEnv in constructor and un-TF obs. space
                                   agent=self.hrl_policy.get_skill_policy(action),
                                   skill_stopping_func=self.hrl_policy.get_skill_stopping_func(action),
                                   max_path_length=self.hrl_policy.skill_max_timesteps,
                                   reset_start_rollout=False  # do not reset the env, continue from current state
                                   )
        # TODO! next_obs will be non-normalized!
        next_obs = AsaEnv.unwrap_to_asa_env(self.env).get_current_obs()
        reward = np.sum(skill_path['rewards'])
        term = skill_path['terminated'][-1]
        return Step(next_obs, reward, term)


    @overrides
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
