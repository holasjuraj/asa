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
        # TODO? terminate if *any* end_obs is reached, or end_obs belonging to start_obs we started from?
        end_obs = self._end_obss[self.current_obs_idx, :]

        # a) Full match
        skill_term = np.array_equiv(obs, end_obs)
        surr_reward = 1 if skill_term else 0

        # # b) Partial match (95/90/80%)
        # skill_term = np.mean(np.abs(obs - end_obs)) < 0.05
        # surr_reward = 1 if skill_term else 0

        # # c) Partial match (90%), reward ~ percentual match
        # diff = np.mean(np.abs(obs - end_obs))
        # skill_term = diff < 0.1
        # surr_reward = 1 - diff


        # DEBUG to store actions and success of this rollout
        self.actions.append(action)
        self.successful = skill_term
        # /DEBUG
        surr_term = term or skill_term
        return obs, surr_reward, surr_term, info

    # DEBUG to log start_obs, end_obs and action of plotted rollout
    def save_rendered_plot(self):
        msg = '\n================================\n'
        msg += 'SUCCESSFUL' if self.successful else 'FAILED\n'
        msg += 'Start observation (with orientation {}°):\n'.format(
            (self.env.env.agent_ori * 180 / np.pi) % 360
        )
        msg += str(self._start_obss[self.current_obs_idx, :].reshape(5, 5)[::-1])
        msg += '\nEnd observation:\n'
        msg += str(self._end_obss[self.current_obs_idx, :].reshape(5, 5)[::-1])

        a = np.array(self.actions)
        msg += '\n\nRaw actions:\n'
        msg += str(a)
        a = np.clip(np.round(a * 1.5), a_min=-1, a_max=1)
        msg += '\n\nDiscretized actions:\n'
        msg += str(a)
        msg += '\n================================'
        logger.log(msg)
        self.env.unwrapped.save_rendered_plot()
    # /DEBUG

    @overrides
    def reset(self, **kwargs):
        self.current_obs_idx = np.random.randint(self._num_obs)
        start_obs = self._start_obss[self.current_obs_idx, :]
        # DEBUG to store actions of this rollout
        self.actions = []
        result = AsaEnv.reset_to_state_wrapped(
                env=self.env,
                start_obs=start_obs,
                **kwargs
        )
        self.start_ori = self.env.env.agent_ori
        return result
        # /DEBUG
        return AsaEnv.reset_to_state_wrapped(
                env=self.env,
                start_obs=start_obs,
                **kwargs
        )
