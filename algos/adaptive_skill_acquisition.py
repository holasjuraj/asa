import os
import numpy as np
import tensorflow as tf
from garage.tf.algos import BatchPolopt
from garage.tf.envs import TfEnv
from garage.tf.policies import CategoricalMLPPolicy
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger
from garage.misc.tensor_utils import flatten_tensors, unflatten_tensors
from sandbox.asa.envs import SkillLearningEnv, HierarchizedEnv
from sandbox.asa.utils import PathTrie
from sandbox.asa.policies import CategoricalMLPSkillIntegrator


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
        self._top_algo_cls = top_algo_cls
        self._top_algo_kwargs = kwargs if kwargs is not None else dict()
        self._low_algo_cls = low_algo_cls
        self._low_algo_kwargs = low_algo_kwargs if low_algo_kwargs is not None else dict()
        self._hrl_policy = hrl_policy
        self._last_f_score = 999999.
        self._added_skills = 0
        self._tf_sess = None

        logger.set_tensorboard_step_key('Iteration')


    @overrides
    def init_opt(self):
        return self._top_algo.init_opt()


    @overrides
    def train(self, sess=None, snapshot_mode=None):
        if sess is None:
            sess = tf.Session()
            sess.__enter__()
        self._tf_sess = sess
        if snapshot_mode is not None:
            logger.set_snapshot_mode(snapshot_mode)
        last_average_return = super(AdaptiveSkillAcquisition, self).train(sess=sess)
        return {
            'last_average_return': last_average_return,
            'snapshot_dir': logger.get_snapshot_dir()
        }


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        res = self._top_algo.get_itr_snapshot(itr, samples_data)
        # TODO? only include path actions and observations to snapshot in order to speed up saving
        res['paths'] = samples_data['paths']  # to be able to construct Trie from exported snapshot
        res['hrl_policy'] = self._hrl_policy
        res['low_algo_cls'] = self._low_algo_cls
        res['low_algo_kwargs'] = self._low_algo_kwargs
        return res


    @overrides
    def optimize_policy(self, itr, samples_data):
        self._top_algo.optimize_policy(itr, samples_data)
        with logger.prefix('ASA | '):
            make_skill, skill_subpath = self.decide_new_skill(samples_data)
            if make_skill:
                new_skill_pol, new_skill_id = self.create_and_train_new_skill(skill_subpath)
                self.integrate_new_skill(new_skill_id, skill_subpath)


    def decide_new_skill(self, samples_data):
        """
        Decide if new skill should be made. If yes, return also start and end observations for training.
        :param samples_data: processed sampled data:
                dict(observations, actions, advantages, rewards, returns, valids, agent_infos, env_infos, paths)
        :return: (bool: make new skill, start_obss, end_obss)
        """
        # TODO extract Trie parameters
        min_length = 2
        max_length = 5
        action_map = None  # {0: 's', 1: 'L', 2: 'R'}
        min_f_score = 2
        max_results = 10
        aggregations = []  # sublist of ['mean', 'most_freq', 'nearest_mean', 'medoid'] or 'all'
        f_score_step_factor = 1.5

        # TODO? share path trie among more batches?
        paths = samples_data['paths']
        path_trie = PathTrie(self._hrl_policy.num_skills)
        for path in paths:
            actions = path['actions'].argmax(axis=1).tolist()
            observations = path['observations']
            path_trie.add_all_subpaths(
                actions,
                observations,
                min_length=min_length,
                max_length=max_length
            )
        logger.log('Searched {} rollouts'.format(len(paths)))

        frequent_paths = path_trie.items(
            action_map=action_map,
            min_count=10,  # len(paths) * 2,   # TODO? what about this?
            min_f_score=min_f_score,
            max_results=max_results,
            aggregations=aggregations
        )
        logger.log('Found {} frequent paths: [index, actions, count, f-score]'.format(len(frequent_paths)))
        for i, f_path in enumerate(frequent_paths):
            logger.log('    {:2}: {:{pad}}\t{}\t{:.3f}'.format(
                i,
                str(f_path['actions']),
                f_path['count'],
                f_path['f_score'],
                pad=max_length*3))

        # return False, None    # DEBUG prevent training of new skill
        if self._added_skills > 0:
            return False, None  # DEBUG add only one skill
        # TODO? some more clever mechanism to decide if we need a new skill?
        #       As-is, we make new skill if its f-score is more than f_score_step_factor - times greater then previous one.
        if len(frequent_paths) == 0:
            return False, None
        top_subpath = frequent_paths[0]
        prev_f_score, self._last_f_score = self._last_f_score, top_subpath['f_score']
        if self._last_f_score > prev_f_score * f_score_step_factor:
            logger.log('Decided to make new skill, since its f-score {} > {} * {}'.format(self._last_f_score, f_score_step_factor, prev_f_score))
            logger.log('New skill is based on subpath: {}'.format(top_subpath['actions']))
            self._added_skills += 1
            return True, top_subpath
        return False, None


    def create_and_train_new_skill(self, skill_subpath):
        """
        Create and train a new skill based on given subpath. The new skill policy and
        ID are returned, and also saved in self._hrl_policy.
        """
        ## Prepare elements for training
        # Environment
        skill_learning_env = TfEnv(
                SkillLearningEnv(
                    # base env that was wrapped in HierarchizedEnv (not fully unwrapped - may be normalized!)
                    env=self.env.env.env,  # TODO how much do we want to unwrap the environment?
                    start_obss=skill_subpath['start_observations'],
                    end_obss=skill_subpath['end_observations']
                )
        )

        # Skill policy
        new_skill_pol, new_skill_id = self._hrl_policy.create_new_skill(skill_subpath['end_observations'])  # blank policy to be trained

        # Baseline - clone baseline specified in low_algo_kwargs, or top-algo`s baseline
        #   We need to clone baseline, as each skill policy must have its own instance
        la_kwargs = dict(self._low_algo_kwargs)
        baseline_to_clone = la_kwargs.get('baseline', self.baseline)
        baseline = Serializable.clone(  # to create blank baseline
                obj=baseline_to_clone,
                name='{}Skill{}'.format(type(baseline_to_clone).__name__, new_skill_id)
        )
        la_kwargs['baseline'] = baseline

        # Algorithm
        algo = self._low_algo_cls(
                env=skill_learning_env,
                policy=new_skill_pol,
                **la_kwargs
        )

        # Logger parameters
        logger.dump_tabular(with_prefix=False)
        logger.log('Launching training of the new skill')
        logger_snapshot_dir_before = logger.get_snapshot_dir()
        logger_snapshot_mode_before = logger.get_snapshot_mode()
        logger_snapshot_gap_before = logger.get_snapshot_gap()
        logger.set_snapshot_dir(os.path.join(
                logger_snapshot_dir_before,
                'skill{}'.format(new_skill_id)
        ))
        logger.set_snapshot_mode('none')
        # logger.set_snapshot_gap(max(1, np.floor(la_kwargs['n_itr'] / 10)))
        logger.push_tabular_prefix('Skill{}/'.format(new_skill_id))
        logger.set_tensorboard_step_key('Iteration')

        # Train new skill
        with logger.prefix('Skill {} | '.format(new_skill_id)):
            algo.train(sess=self._tf_sess)

        # Restore logger parameters
        logger.pop_tabular_prefix()
        logger.set_snapshot_dir(logger_snapshot_dir_before)
        logger.set_snapshot_mode(logger_snapshot_mode_before)
        logger.set_snapshot_gap(logger_snapshot_gap_before)
        logger.log('Training of the new skill finished')

        return new_skill_pol, new_skill_id


    def integrate_new_skill(self, new_skill_id, new_skill_subpath):
        # TODO? extract integrator parameter
        skill_integration_method = CategoricalMLPSkillIntegrator.Method.SUBPATH_SKILLS_AVG

        ## Hierarchized environment
        hrl_env = HierarchizedEnv(
                # base env that was wrapped in HierarchizedEnv (not fully unwrapped - may be normalized!)
                env=self.env.env.env,   # TODO how much do we want to unwrap the environment?
                num_orig_skills=self._hrl_policy.num_skills
        )
        tf_hrl_env = TfEnv(hrl_env)

        ## Top policy
        # 1) Get old policy from saved data
        old_top_policy = self._hrl_policy.get_top_policy()

        # 2) Get weights of old top policy
        otp_weights = unflatten_tensors(
                old_top_policy.get_param_values(),
                old_top_policy.get_param_shapes()
        )

        # 3) Create weights for new top policy
        skill_integrator = CategoricalMLPSkillIntegrator()
        ntp_weight_values = skill_integrator.integrate_skill(
                old_policy_weights=otp_weights,
                method=skill_integration_method,
                # Specific parameters for START_OBSS_SKILLS_AVG
                subpath_start_obss=new_skill_subpath['start_observations'],
                top_policy=old_top_policy,
                # Specific parameters for SUBPATH_SKILLS_AVG, SUBPATH_SKILLS_SMOOTH_AVG and SUBPATH_FIRST_SKILL
                subpath_actions=new_skill_subpath['actions']
        )

        # 4) Create new policy and randomly initialize its weights
        new_top_policy = CategoricalMLPPolicy(
                env_spec=tf_hrl_env.spec,  # This env counts with new skill (action space = n + 1)
                hidden_sizes=(32, 32),     # As was in asa_test.py,
                name='CategoricalMLPPolicyWithSkill{}'.format(new_skill_id)
        )
        ntp_init_op = tf.variables_initializer(new_top_policy.get_params())
        ntp_init_op.run()

        # 5) Fill new policy with adjusted weights
        new_top_policy.set_param_values(
                flattened_params=flatten_tensors(ntp_weight_values)
        )

        ## Adjust HRL policy and training algorithms
        self._hrl_policy.top_policy = new_top_policy
        hrl_env.set_hrl_policy(self._hrl_policy)
        self.env = tf_hrl_env
        self.policy=self._hrl_policy.get_top_policy()
        self._top_algo = self._top_algo_cls(
                env=tf_hrl_env,
                policy=self._hrl_policy.get_top_policy(),
                baseline=self.baseline,
                **self._top_algo_kwargs
        )
        self.sampler = self._top_algo.sampler
        self._top_algo.init_opt()
        self.start_worker(self._tf_sess)
