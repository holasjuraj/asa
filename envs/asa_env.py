from abc import ABC

from gym import Env

from garage.core.serializable import Serializable
from garage.tf.envs import TfEnv
from garage.envs.normalized_env import NormalizedEnv
from garage.misc import logger


warned_msgs = set()


def warn_once(text):
    global warned_msgs
    if text in warned_msgs:
        return
    warned_msgs.add(text)
    logger.log(text)


class AsaEnv(Env, ABC):
    """
    Abstract class that must be inherited by an environment to support ASA framework.
    """

    def get_current_obs(self):
        """
        Obtain current observation without changing state
        """
        raise NotImplementedError


    def reset_to_state(self, start_obs, **kwargs):
        """
        Reset the environment, but start in a state that will yield start_obs as initial observation.
        :param start_obs: desired initial observation
        """
        raise NotImplementedError


    @staticmethod
    def unwrap_to_asa_env(env):
        """
        Unwrap environment from wrapper/s to reveal original AsaEnv environment.
        """
        while not isinstance(env, AsaEnv):
            env = env.env
        return env

    @staticmethod
    def get_current_obs_wrapped(env):
        """
        Perform get_current_obs on AsaEnv wrapped into multiple other environment wrappers.
        After receiving observation, apply wrappers` logic upon it.
        Supported wrappers:
        - TfEnv: no change to observation
        - NormalizedEnv: apply obs normalization if needed
        - other: no change, display warning
        :param env: AsaEnv wrapped in multiple other environment wrappers
        """
        # Unwrap
        stack = []
        while not isinstance(env, AsaEnv):
            stack.append((type(env), env))
            env = env.env
        # Get obs
        obs = env.get_current_obs()
        # Re-wrap
        while stack:
            wrapper_cls, wrapper_env = stack.pop()
            if wrapper_cls is TfEnv:
                # No action needed
                pass
            elif wrapper_cls is NormalizedEnv:
                if wrapper_env._normalize_obs:
                    obs = wrapper_env._apply_normalize_obs(obs)
            else:
                warn_once(
                    'AsaEnv: get_current_obs_wrapped performed on unknown wrapper "{}". '
                    'Effect of this wrapper was not applied!'.format(wrapper_cls))
        return obs

    @staticmethod
    def reset_to_state_wrapped(env, start_obs, **kwargs):
        """
        Perform reset_to_state on AsaEnv wrapped into multiple other environment wrappers.
        After receiving initial observation, apply wrappers` logic upon it.
        Supported wrappers:
        - TfEnv: no change to observation
        - NormalizedEnv: apply obs normalization if needed
        - other: no change, display warning
        :param env: AsaEnv wrapped in multiple other environment wrappers
        :param start_obs: desired initial observation
        """
        # Unwrap
        stack = []
        while not isinstance(env, AsaEnv):
            stack.append((type(env), env))
            env = env.env
        # Reset
        obs = env.reset_to_state(start_obs, **kwargs)
        # Re-wrap
        while stack:
            wrapper_cls, wrapper_env = stack.pop()
            if wrapper_cls is TfEnv:
                # No action needed
                pass
            elif wrapper_cls is NormalizedEnv:
                if wrapper_env._normalize_obs:
                    obs = wrapper_env._apply_normalize_obs(obs)
            else:
                warn_once(
                    'AsaEnv: reset_to_state_wrapped performed on unknown wrapper "{}". '
                    'Effect of this wrapper was not applied!'.format(wrapper_cls))
        return obs

    @staticmethod
    def clone_wrapped(env):
        """
        Clone Serializable AsaEnv wrapped into multiple other environment wrappers.
        This performs Serializable.clone on inner env and all its wrappers, if possible.
        Supported wrappers:
        - TfEnv: no cloning needed, only wrap in new instance
        - NormalizedEnv: clone and wrap
        - other Serializable: clone and wrap, display warning
        - other non-Serializable: wrap in new instance, display warning
        :param env: AsaEnv wrapped in multiple other environment wrappers
        """
        # Unwrap
        stack = []
        while not isinstance(env, AsaEnv):
            stack.append((type(env), env))
            env = env.env
        # Clone inner env
        new_env = Serializable.clone(env)
        # Re-wrap, cloning wrappers on the way
        while stack:
            wrapper_cls, wrapper_env = stack.pop()
            if wrapper_cls is TfEnv:
                new_env = TfEnv(new_env)
            elif wrapper_cls is NormalizedEnv:
                # WARNING: obs_mean and obs_var are not copied to original env after skill training!
                new_env = Serializable.clone(wrapper_env, env=new_env)
            elif isinstance(wrapper_env, Serializable):
                new_env = Serializable.clone(wrapper_env, env=new_env)
                warn_once(
                    'AsaEnv: clone_wrapped performed on unknown Serializable wrapper "{}". '
                    'Wrapper was cloned and applied.'.format(wrapper_cls))
            else:
                new_env = wrapper_cls(env=new_env)
                warn_once(
                    'AsaEnv: clone_wrapped performed on unknown non-Serializable wrapper "{}". '
                    'Wrapper was initiated with default parameters and applied.'.format(wrapper_cls))
        return new_env
