import numpy as np

from gym import Wrapper
from gym.spaces import Discrete
from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from garage.envs.base import Step
from garage.tf.envs import TfEnv
from garage.tf.misc.tensor_utils import concat_tensor_dict_list

from sandbox.asa.sampler import skill_rollout
from sandbox.asa.envs import AsaEnv


class HierarchizedEnv(Wrapper, Serializable):
    def __init__(
            self,
            env,
            num_orig_skills,
            subpath_infos=None
    ):
        """
        Creates a top-level environment for a HRL agent. Original env`s actions are replaced by N discrete actions,
        N being the number of skills.
        :param env: AsaEnv environment to wrap
        :param num_orig_skills: number of pre-trained skill that will prepared be in HRL policy
        :param subpath_infos: 'all' or list of subpath information to keep, defaults to ['env_infos']
        """
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self._num_orig_skills = num_orig_skills
        self.action_space = Discrete(self._num_orig_skills)
        self.hrl_policy = None
        if subpath_infos is None:
            subpath_infos = ['env_infos']
        self.subpath_infos = subpath_infos


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
        skill_path = skill_rollout(env=TfEnv(self.env),
                                   agent=self.hrl_policy.get_skill_policy(action),
                                   skill_stopping_func=self.hrl_policy.get_skill_stopping_func(action),
                                   max_path_length=self.hrl_policy.skill_max_timesteps,
                                   reset_start_rollout=False  # do not reset the env, continue from current state
                                   )
        next_obs = AsaEnv.get_current_obs_wrapped(self.env)
        reward = np.sum(skill_path['rewards'])
        term = skill_path['terminated'][-1]

        return Step(
            observation=next_obs,
            reward=reward,
            done=term,
            subpath_infos=SubpolicyPathInfo(skill_path, store=self.subpath_infos)
        )


    @overrides
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class SubpolicyPathInfo:
    """
    Container class to hold path info from subpolicy rollouts.
    """
    def __init__(self, path, store='all'):
        """
        :param path: path dict containing all the information
        :param store: list of information to store
        :type store: str or list
        """
        self.as_dict = dict()
        if store == 'all':
            store = ['observations', 'actions', 'rewards', 'env_infos',
                     'advantages', 'deltas', 'baselines', 'returns']
        for thing in store:
            self.as_dict[thing] = path.get(thing, np.array([]))

    def __getitem__(self, item):
        return self.as_dict[item]

    def __repr__(self):
        return repr(self.as_dict)

    @staticmethod
    def concat_subpath_infos(infos_list):
        """
        Concat information from list<SubpolicyPathInfo> into dict
        """
        return concat_tensor_dict_list([info_obj.as_dict for info_obj in infos_list])
