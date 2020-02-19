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
        new_weights = [np.copy(value) for value in old_policy_weights]
        out_w = new_weights[-2]
        out_b = new_weights[-1]

        if method is self.Method.RANDOM:
            skill_w, skill_b = self._weights_random(out_w, out_b, **kwargs)
        elif method is self.Method.RANDOM_BIASED:
            skill_w, skill_b = self._weights_random_biased(out_w, out_b, **kwargs)
        elif method is self.Method.START_OBSS_SKILLS_AVG:
            skill_w, skill_b = self._weights_start_obss_skills_avg(out_w, out_b, **kwargs)
        elif method is self.Method.SUBPATH_SKILLS_AVG:
            skill_w, skill_b = self._weights_subpath_skills_avg(out_w, out_b, **kwargs)
        elif method is self.Method.SUBPATH_SKILLS_SMOOTH_AVG:
            skill_w, skill_b = self._weights_subpath_skills_smooth_avg(out_w, out_b, **kwargs)
        elif method is self.Method.SUBPATH_FIRST_SKILL:
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
        num_actions = old_w.shape[1]
        action_counts = [np.sum(subpath_actions == a) for a in range(num_actions)]
        w, b = self._get_weighted_average(old_w, old_b, avg_weights=action_counts)
        return w, b

    def _weights_subpath_skills_smooth_avg(self, old_w, old_b, subpath_actions, **kwargs):
        # TODO implement, kwargs = subpath['actions']
        w = 0
        b = 0
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
