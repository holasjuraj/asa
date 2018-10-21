from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv


class SkillExecutionEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env
    ):
        """
        Creates an environment executing a single (learnt) skill. Trajectory is terminated upon reaching end_obs state.
        :param env: Environment to wrap
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
