from rllab.core.serializable import Serializable
from rllab.spaces import Discrete


class HierarchicalPolicy(Serializable):
    """
    Wrapper for hierarchical policy containing top-level and skill policies.
    """

    # noinspection PyMissingConstructor
    def __init__(
            self,
            env_spec,
            top_policy,
            skill_policy_prototype,
            skill_policies,
            skill_max_timesteps
    ):
        Serializable.quick_init(self, locals())
        self._env_spec = env_spec
        self._top_policy = top_policy
        self._skill_policy_prototype = skill_policy_prototype
        self._skill_policies = skill_policies
        self.skill_max_timesteps = skill_max_timesteps
        # Check top-level policy
        if not isinstance(top_policy.action_space, Discrete) \
                or top_policy.action_space.n != self.num_skills:
            raise TypeError('Top level policy must have Discrete(num_skills) action space.')

    @property
    def num_skills(self):
        return len(self._skill_policies)

    def get_top_policy(self):
        """
        :return: Policy of top-level agent
        :rtype: rllab.policies.base.Policy
        """
        return self._top_policy

    def get_skill_policy(self, i):
        """
        :param i: Number of skill
        :return: Policy of selected skill
        :rtype: rllab.policies.base.Policy
        """
        return self._skill_policies[i]
