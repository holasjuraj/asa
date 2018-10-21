from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv


class HierarchizedEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env
    ):
        """
        Creates a top-level environment for a HRL agent. Original env`s actions are replaced by N discrete actions,
        N being the number of skills.
        :param env: Environment to wrap
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
