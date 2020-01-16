import numpy as np

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from gym import Wrapper


class SkillLearningEnv(Wrapper, Serializable):
    def __init__(
            self,
            env,
            start_obss,
            end_obss
    ):
        """
        Creates an environment tailored to train a single (missing) skill. Trajectories are initialized in start_obss
        state and terminated (and reward is generated) upon reaching end_obs state.
        :param env: AsaEnv environment to wrap. Environment is cloned to sustain integrity of original env.
        :param start_obss: Tensor of experienced starting observations (where skill should initiate)
        :param end_obss: Tensor of experienced ending observations (where skill should terminate)
        """
        Serializable.quick_init(self, locals())
        # TODO check that this actually initializes cloned env. Un- and re-wrapping to TfEnv might be needed.
        Wrapper.__init__(self, Serializable.clone(env))
        if start_obss.shape != end_obss.shape:
            raise ValueError('start_obss ({}) and end_obss ({}) must be of same shape'
                             .format(start_obss.shape, end_obss.shape))
        self._end_obss = end_obss.reshape((end_obss.shape[0], -1))
        self._start_obss = start_obss.reshape((start_obss.shape[0], -1))

    @property
    def _num_obs(self):
        return self._start_obss.shape[0]

    @property
    def _obs_dim(self):
        return self._start_obss.shape[-1]

    @overrides
    def step(self, action):
        obs, _, term, info = self.env.step(action)
        # TODO terminate if *any* end_obs is reached, or end_obs belonging to start_obs we started from?
        skill_term = obs in self._end_obss
        surr_reward = 1 if skill_term else 0
        surr_term = term or skill_term
        return obs, surr_reward, surr_term, info

    @overrides
    def reset(self, **kwargs):
        start_obs = self._start_obss[np.random.randint(self._num_obs), :]
        return self.env.reset_to_state(start_obs=start_obs, **kwargs)  # TODO wrapping issue: env will be AsaEnv wrapped inside TfEnv/NormalizedEnv
