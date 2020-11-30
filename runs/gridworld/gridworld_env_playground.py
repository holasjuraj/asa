#!/usr/bin/env python

from sandbox.asa.envs import GridworldGathererEnv
from sandbox.asa.policies import GridworldTargetPolicy
from sandbox.asa.sampler import skill_rollout

from garage.tf.envs import TfEnv
from garage.misc import logger



env = TfEnv(GridworldGathererEnv(
        plot={
            'visitation': {
                'save': False,
                'live': True,
                'alpha': 1
            }
        }
))
policy = GridworldTargetPolicy(
        env_spec=env.spec,
        target=(3, 5)
)

logger.set_snapshot_dir('/home/h/holas3/garage/data/local/gridworld-env-playground')
path = skill_rollout(
        env=env,
        agent=policy,
        skill_stopping_func=(lambda *_: False),
        max_path_length=20,
        reset_start_rollout=True
)

env.unwrapped.log_diagnostics([path])
input('< Press Enter to quit >')  # Prevent plots from closing
