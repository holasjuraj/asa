import numpy as np
from abc import ABC

from enum import Enum


class SkillIntegrator(ABC):

    def __init__(self):
        pass

    def integrate_skill(self, old_policy_weights, **kwargs):
        raise NotImplementedError



class CategoricalMLPSkillIntegrator(SkillIntegrator):
    # noinspection PyClassHasNoInit
    class Method(Enum):
        RANDOM = 'random'
        # Initialize weights and bias randomly from normal distribution,
        # keeping mean and std.

        RANDOM_BIASED = 'random_biased'
        # Initialize weights randomly from normal distribution, keeping mean
        # and std. Make bias significantly greater to encourage exploration of
        # new sill.

        OLD_SKILL_AVG = 'old_skills_avg'
        # Initialize weights as weighted average of old skills` weights.
        # Average is weighted w.r.t. frequency of choosing an old skill in
        # situations with new_skill.start_obss observation.


    def integrate_skill(self, old_policy_weights, method=Method.RANDOM, **kwargs):
        new_weights = [np.copy(value) for value in old_policy_weights]
        out_w = new_weights[-2]
        out_b = new_weights[-1]

        if method is self.Method.RANDOM:
            skill_w, skill_b = self._weights_random(out_w, out_b)
        elif method is self.Method.RANDOM_BIASED:
            skill_w, skill_b = self._weights_random_biased(out_w, out_b)
        elif method is self.Method.OLD_SKILL_AVG:
            skill_w, skill_b = self._weights_old_skill_avg(out_w, out_b)
        else:
            raise NotImplementedError

        skill_w = skill_w.reshape((out_w.shape[0], 1))
        skill_b = skill_b.reshape((1,))
        new_out_w = np.concatenate([out_w, skill_w], axis=1)
        new_out_b = np.concatenate((out_b, skill_b))
        new_weights[-2] = new_out_w
        new_weights[-1] = new_out_b
        return new_weights

    @staticmethod
    def _weights_random(old_w, old_b):
        w = np.random.normal(size=old_w.shape[0], loc=np.mean(old_w), scale=np.std(old_w))
        b = np.random.normal(size=1, loc=np.mean(old_b), scale=np.std(old_b))
        return w, b

    @staticmethod
    def _weights_random_biased(old_w, old_b):
        w = np.random.normal(size=old_w.shape[0], loc=np.mean(old_w), scale=np.std(old_w))
        b = np.array([np.max(old_b) + np.ptp(old_b)])
        return w, b

    @staticmethod
    def _weights_old_skill_avg(old_w, old_b):
        # DEBUG only working for MinibotRightPolicy as a new skill - taking
        #       'left' weights and copying them as 'right' weights (+ noise)
        w = np.copy(old_w[:, 1]) + np.random.normal(size=old_w.shape[0], scale=0.01)
        b = np.copy(old_b[1]) + np.random.normal(size=1, scale=0.01)
        return w, b
