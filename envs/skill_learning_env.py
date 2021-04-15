import numpy as np

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from gym import Wrapper
from garage.misc import logger

from sandbox.asa.envs import AsaEnv


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
        Wrapper.__init__(self, AsaEnv.clone_wrapped(env))  # this clones base env along with all wrappers
        if start_obss.shape != end_obss.shape:
            raise ValueError('start_obss ({}) and end_obss ({}) must be of same shape'
                             .format(start_obss.shape, end_obss.shape))
        self._end_obss = end_obss.reshape((end_obss.shape[0], -1))
        self._start_obss = start_obss.reshape((start_obss.shape[0], -1))
        self.current_obs_idx = None

    @property
    def _num_obs(self):
        return self._start_obss.shape[0]

    @property
    def _obs_dim(self):
        return self._start_obss.shape[-1]

    @overrides
    def step(self, action):
        obs, _, term, info = self.env.step(action)
        # Terminate if agent reached end_obs belonging to start_obs it started from
        end_obs = self._end_obss[self.current_obs_idx, :]

        # # a) Full match
        # skill_term = np.array_equiv(obs, end_obs)
        # surr_reward = 1 if skill_term else 0

        # # b) Partial match (95/90/80%)
        # skill_term = np.mean(np.abs(obs - end_obs)) < 0.05
        # surr_reward = 1 if skill_term else 0

        # # c) Partial match (90%), reward ~ percentual match
        # diff = np.mean(np.abs(obs - end_obs))
        # skill_term = diff < 0.1
        # surr_reward = 1 - diff

        # d) Per-dimension partial match
        threshold = 0.1
        skill_term = np.max(np.abs(obs - end_obs)) < threshold
        surr_reward = 1 if skill_term else 0

        surr_term = term or skill_term
        return obs, surr_reward, surr_term, info


    @overrides
    def reset(self, **kwargs):
        self.current_obs_idx = np.random.randint(self._num_obs)
        start_obs = self._start_obss[self.current_obs_idx, :]
        return AsaEnv.reset_to_state_wrapped(
                env=self.env,
                start_obs=start_obs,
                **kwargs
        )
