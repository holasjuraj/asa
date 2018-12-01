from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv


class SkillExecutionEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env
    ):
        """
        ABORTED: Functionality is replaced by sampler.utils.skill_rollout() .
        REASON: This environment would override (thus hide) wrapped env`s termination signal, which is required to be
                propagated to HierarchizedEnv.step() and further "out".

        Creates an environment executing a single (learnt) skill. Trajectory is terminated upon reaching end_obs state.
        :param env: Environment to wrap
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
