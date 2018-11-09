from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc.overrides import overrides
from sandbox.asa.envs.skill_learning_env import SkillLearningEnv


class AdaptiveSkillAcquisition(BatchPolopt):

    def __init__(self,
                 env,
                 hrl_policy,
                 baseline,
                 top_algo_cls,
                 top_algo_kwargs,
                 low_algo_cls,
                 low_algo_kwargs,
                 **kwargs):
        """
        Wrapper for a top-level RL algorithm that performs Adaptive Skill Acquisition in HRL.
        :param env: hierarchized environment
        :type env: HierarchizedEnv
        :param hrl_policy: hierarchy of policies, including (blank) top-level policy that will be trained, and a set of
                           pre-trained skill policies. ASA might add new skills to this set.
        :type hrl_policy: HierarchicalPolicy
        :param baseline: baseline
        :param top_algo_cls: class of RL algorithm for training top-level agent. Must inherit BatchPolopt (only
                             init_opt(), optimize_policy(), and get_itr_snapshot() will be used).
        :param top_algo_kwargs: additional kwargs for top-level algorithm (don`t have to provide env, policy, baseline)
        :param low_algo_cls: class of RL algorithm for training low-level agents - each new skill.
        :param low_algo_kwargs: additional kwargs for low-level algorithm (don`t have to provide env, policy, baseline)
        """
        super().__init__(env,
                         hrl_policy.get_top_policy(),
                         baseline,
                         **kwargs)
        self._top_algo = top_algo_cls(env,
                                      hrl_policy.get_top_policy(),
                                      baseline,
                                      **top_algo_kwargs)
        self._low_algo_cls = low_algo_cls
        self._low_algo_kwargs = low_algo_kwargs
        self._hrl_policy = hrl_policy
        # TODO assertion for types

    @overrides
    def init_opt(self):
        self._top_algo.init_opt()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        res = self._top_algo.get_itr_snapshot(itr, samples_data)
        # TODO res['some hrl stuff?'] = None
        return res

    @overrides
    def optimize_policy(self, itr, samples_data):
        self._top_algo.optimize_policy(itr, samples_data)
        # TODO change pseudo-code to actual code:
        if self.decide_new_skill(samples_data):
            start_obss, end_obss = self.extract_start_end_obss(samples_data)
            self.train_new_skill(start_obss, end_obss)

    def decide_new_skill(self, samples_data):
        # TODO
        pass

    def extract_start_end_obss(self, samples_data):
        # TODO
        pass

    def train_new_skill(self, start_obss, end_obss):
        # TODO
        new_skill_pol = self._hrl_policy.create_new_skill()  # blank policy to be trained
        baseline = self._low_algo_kwargs.get('baseline', self.baseline)
        # baseline = type(baseline)(...)  # TODO! change to fresh baseline
        self._low_algo_kwargs['baseline'] = self.baseline

        algo = self._low_algo_cls(env=SkillLearningEnv(env=self.env._wrapped_env,
                                                       start_obss=start_obss,
                                                       end_obss=end_obss),
                                  policy=new_skill_pol,
                                  **self._low_algo_kwargs)
