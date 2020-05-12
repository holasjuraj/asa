import numpy as np
from garage.core import Serializable
from garage.policies import Policy


class MinibotForwardPolicy(Policy, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        return np.array([1, 1]), dict()

    def get_params_internal(self, **tags):
        return []



class MinibotLeftPolicy(Policy, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        return np.array([-1, 1]), dict()

    def get_params_internal(self, **tags):
        return []



class MinibotRightPolicy(Policy, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        return np.array([1, -1]), dict()

    def get_params_internal(self, **tags):
        return []



class MinibotRandomPolicy(Policy, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        return np.random.rand(2) * 2 - 1, dict()

    def get_params_internal(self, **tags):
        return []
