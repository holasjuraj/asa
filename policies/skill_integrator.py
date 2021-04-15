import numpy as np
from abc import ABC

from enum import Enum


class SkillIntegrator(ABC):

    def __init__(self):
        pass

    def integrate_skill(self, old_policy_weights, **kwargs):
        raise NotImplementedError


# noinspection PyMethodMayBeStatic
# noinspection PyClassHasNoInit
class CategoricalMLPSkillIntegrator(SkillIntegrator):
    class Method(Enum):
        RANDOM = 'rnd'
        # Initialize weights and bias randomly from normal distribution,
        # keeping mean and std.

        RANDOM_BIASED = 'rndBias'
        # Initialize weights randomly from normal distribution, keeping mean
        # and std. Make bias significantly greater to encourage exploration of
        # new sill.

        START_OBSS_SKILLS_AVG = 'startObsAvg'
        # Initialize weights as weighted average of old skills` weights.
        # Average is weighted w.r.t. frequency of choosing an old skill in
        # situations with new_skill.start_obss observation.

        SUBPATH_SKILLS_AVG = 'sbptAvg'
        # Initialize weights as weighted average of old skills` weights.
        # Average is weighted w.r.t. frequency of using an old skill in
        # subpath that defines new skill.

        SUBPATH_SKILLS_SMOOTH_AVG = 'sbptSmthAvg'
        # Initialize weights as weighted average of old skills` weights.
        # New weights are computed as exponentially smoothed average of skills
        # used in subpath that defines new skill (first skill=step has greatest
        # weight in average, last has lowest).

        SUBPATH_FIRST_SKILL = 'sbptFrst'
        # Initialize weights by copying weights of skill that was used as first
        # step of subpath that defines new skill. Small noise is added to
        # prevent the two skills to be chosen identically.


    def integrate_skill(self, old_policy_weights, method=Method.RANDOM, **kwargs):
        if type(method) is not str:
            method = method.value
        new_weights = [np.copy(value) for value in old_policy_weights]
        out_w = new_weights[-2]
        out_b = new_weights[-1]

        if method == self.Method.RANDOM.value:
            skill_w, skill_b = self._weights_random(out_w, out_b, **kwargs)
        elif method == self.Method.RANDOM_BIASED.value:
            skill_w, skill_b = self._weights_random_biased(out_w, out_b, **kwargs)
        elif method == self.Method.START_OBSS_SKILLS_AVG.value:
            skill_w, skill_b = self._weights_start_obss_skills_avg(out_w, out_b, **kwargs)
        elif method == self.Method.SUBPATH_SKILLS_AVG.value:
            skill_w, skill_b = self._weights_subpath_skills_avg(out_w, out_b, **kwargs)
        elif method == self.Method.SUBPATH_SKILLS_SMOOTH_AVG.value:
            skill_w, skill_b = self._weights_subpath_skills_smooth_avg(out_w, out_b, **kwargs)
        elif method == self.Method.SUBPATH_FIRST_SKILL.value:
            skill_w, skill_b = self._weights_subpath_first_skill(out_w, out_b, **kwargs)
        else:
            raise NotImplementedError

        skill_w = skill_w.reshape((out_w.shape[0], 1))
        skill_b = skill_b.reshape((1,))
        new_out_w = np.concatenate([out_w, skill_w], axis=1)
        new_out_b = np.concatenate((out_b, skill_b))
        new_weights[-2] = new_out_w
        new_weights[-1] = new_out_b
        return new_weights


    def _weights_random(self, old_w, old_b, **kwargs):
        w = np.random.normal(size=old_w.shape[0], loc=np.mean(old_w), scale=np.std(old_w))
        b = np.random.normal(size=1, loc=np.mean(old_b), scale=np.std(old_b))
        return w, b

    def _weights_random_biased(self, old_w, old_b, **kwargs):
        w = np.random.normal(size=old_w.shape[0], loc=np.mean(old_w), scale=np.std(old_w))
        b = np.array([np.max(old_b) + np.ptp(old_b)])
        return w, b

    def _weights_start_obss_skills_avg(self, old_w, old_b, subpath_start_obss, top_policy, **kwargs):
        action_probs = []
        for obs in subpath_start_obss:
            _, ag_info = top_policy.get_action(obs)  # we toss away the actual action
            action_probs.append(ag_info['prob'])
        action_probs = np.array(action_probs)
        mean_action_probs = np.mean(action_probs, axis=0)
        w, b = self._get_weighted_average(old_w, old_b, avg_weights=mean_action_probs)
        return w, b

    def _weights_subpath_skills_avg(self, old_w, old_b, subpath_actions, **kwargs):
        subpath_actions = np.asarray(subpath_actions)
        num_actions = old_w.shape[1]
        action_counts = [np.sum(subpath_actions == a) for a in range(num_actions)]
        w, b = self._get_weighted_average(old_w, old_b, avg_weights=action_counts)
        return w, b

    def _weights_subpath_skills_smooth_avg(self, old_w, old_b, subpath_actions, **kwargs):
        subpath_actions = np.asarray(subpath_actions)
        last_action_weight = 0.01  # may be tuned
        num_actions = old_w.shape[1]
        subpath_length = len(subpath_actions)
        q = np.power(last_action_weight, 1 / (subpath_length - 1))  # 1 * q^(num_actions-1) == last_action_weight
        qs = np.ones(subpath_length)
        qs[1:] = q
        qs = np.cumprod(qs)
        action_weights = [np.sum(qs[subpath_actions == a]) for a in range(num_actions)]
        w, b = self._get_weighted_average(old_w, old_b, avg_weights=action_weights)
        return w, b

    def _weights_subpath_first_skill(self, old_w, old_b, subpath_actions, **kwargs):
        a = subpath_actions[0]
        w = np.copy(old_w[:, a]) + np.random.normal(size=old_w.shape[0], scale=0.01)
        b = np.copy(old_b[a]) + np.random.normal(size=1, scale=0.01)
        return w, b


    def _get_weighted_average(self, old_w, old_b, avg_weights):
        avg_weights /= np.sum(avg_weights)
        w = old_w @ avg_weights
        b = old_b @ avg_weights
        return w, b

    @staticmethod
    def get_method_str_by_index(idx):
        return list(CategoricalMLPSkillIntegrator.Method)[idx].value

    @staticmethod
    def get_index_of_method_str(method_str):
        return {m.value: idx
                for idx, m in enumerate(list(CategoricalMLPSkillIntegrator.Method))
               }[method_str]
