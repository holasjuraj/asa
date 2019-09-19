from garage.tf.algos import BatchPolopt
from garage.tf.envs import TfEnv
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger
from sandbox.asa.envs import SkillLearningEnv
from sandbox.asa.utils import PathTrie


class AdaptiveSkillAcquisition(BatchPolopt):

    def __init__(self,
                 env,
                 hrl_policy,
                 baseline,
                 top_algo_cls,
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
        :param low_algo_cls: class of RL algorithm for training low-level agents - each new skill.
        :param low_algo_kwargs: additional kwargs for low-level algorithm (don`t have to provide env, policy, baseline)
        """
        # We must init _top_algo before super().__init__, because super().__init__ calls init_opt(),
        # which calls _top_algo.init_opt().
        self._top_algo = top_algo_cls(env=env,
                                      policy=hrl_policy.get_top_policy(),
                                      baseline=baseline,
                                      **kwargs)
        super().__init__(env=env,
                         policy=hrl_policy.get_top_policy(),
                         baseline=baseline,
                         **kwargs)
        self.sampler = self._top_algo.sampler
        self._low_algo_cls = low_algo_cls
        self._low_algo_kwargs = low_algo_kwargs if low_algo_kwargs is not None else dict()
        self._hrl_policy = hrl_policy
        # TODO assertion for types

    @overrides
    def init_opt(self):
        return self._top_algo.init_opt()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        res = self._top_algo.get_itr_snapshot(itr, samples_data)
        # Include 'algo' so that we can run resume_experiment.py
        res['algo'] = self
        # TODO? res['some hrl stuff'] = None
        return res

    @overrides
    def optimize_policy(self, itr, samples_data):
        self._top_algo.optimize_policy(itr, samples_data)
        with logger.prefix('ASA | '):
            make_skill, start_obss, end_obss = self.decide_new_skill(samples_data)
            if make_skill:
                self.make_new_skill(start_obss, end_obss)
                # TODO integrate skill into top-level policy

    def decide_new_skill(self, samples_data):
        """
        Decide if new skill should be made. If yes, return also start and end observations for training.
        :param samples_data: processed sampled data:
                dict(observations, actions, advantages, rewards, returns, valids, agent_infos, env_infos, paths)
        :return: (bool: make new skill, start_obss, end_obss)
        """
        # TODO extract Trie parameters
        min_length = 3
        max_length = 10
        action_map = {0: 's', 1: 'L'}
        min_f_score = 2
        max_results = 10
        aggregations = ['mean']  # sublist of ['mean', 'most_freq', 'nearest_mean', 'medoid'] or 'all'

        # TODO? share path trie among more batches?
        paths = samples_data['paths']
        path_trie = PathTrie(self._hrl_policy.num_skills)
        for path in paths:
            actions = path['actions'].argmax(axis=1).tolist()
            observations = path['observations']
            path_trie.add_all_subpaths(actions,
                                       observations,
                                       min_length=min_length,
                                       max_length=max_length)
        logger.log('Searched {} rollouts'.format(len(paths)))

        frequent_paths = path_trie.items(
            action_map=action_map,
            min_count=10,  # len(paths) * 2,   # TODO? what about this?
            min_f_score=min_f_score,
            max_results=max_results,
            aggregations=aggregations
        )
        logger.log('Found {} frequent paths: [actions, count f-score]'.format(len(frequent_paths)))
        for f_path in frequent_paths:
            logger.log('    {:{pad}}\t{}\t{:.3f}'.format(
                f_path['actions_text'],
                f_path['count'],
                f_path['f_score'],
                pad=max_length))

        # TODO? some more clever mechanism to decide if we need a new skill.
        # As-is, we take the subpath with highest f-score if it is grater then min_f_score. If no such subpath was
        # found, then no skill is created.
        # Hence Trie parameters should be max_results = 1, min_f_score = <some reasonably high number, e.g. 20>
        if len(frequent_paths) == 0:
            return False, None, None
        return False, None, None  # TODO delete me (debug)
        top_subpath = frequent_paths[0]
        return True, top_subpath.start_observations, top_subpath.end_observations

    def make_new_skill(self, start_obss, end_obss):
        """
        Create and train a new skill based on given start and end observations
        """
        new_skill_pol, new_skill_id = self._hrl_policy.create_new_skill(end_obss)  # blank policy to be trained

        learning_env = TfEnv(
                        SkillLearningEnv(
                            # base env that was wrapped in HierarchizedEnv (not fully unwrapped - may be normalized!)
                            env=self.env.env,
                            start_obss=start_obss,
                            end_obss=end_obss
                        )
        )

        la_kwargs = dict(self._low_algo_kwargs)
        # We need to clone baseline, as each skill policy must have its own instance
        baseline_to_clone = la_kwargs.get('baseline', self.baseline)
        baseline = Serializable.clone(baseline_to_clone)    # to create blank baseline
        la_kwargs['baseline'] = baseline

        algo = self._low_algo_cls(env=learning_env,
                                  policy=new_skill_pol,
                                  **la_kwargs)

        with logger.prefix('Skill {} | '.format(new_skill_id)):
            algo.train()
