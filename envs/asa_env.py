from abc import ABC

from gym import Env
from garage.misc.overrides import overrides


class AsaEnv(Env, ABC):
    """
    Abstract class that must be inherited by an environment to support ASA framework.
    """

    def get_current_obs(self):
        """
        Obtain current observation without changing state
        """
        raise NotImplementedError

    @overrides
    def reset_to_state(self, start_obs, **kwargs):
        """
        Reset the environment, but start in a state that will yield start_obs as initial observation.
        :param start_obs: desired initial observation
        """
        raise NotImplementedError
