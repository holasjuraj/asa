import numpy as np
import time
from garage.misc import tensor_utils


def skill_rollout(env,
                  agent,
                  max_path_length=np.inf,
                  skill_stopping_func=None,
                  reset_start_rollout=True,
                  keep_rendered_rgbs=False,
                  animated=False,
                  speedup=1
                  ):
    """
    Perform one rollout in given environment.
    Code adopted from https://github.com/florensacc/snn4hrl
    :param env: AsaEnv environment to run in
    :param agent: Policy to sample actions from
    :param max_path_length: force terminate the rollout after this many steps
    :param skill_stopping_func: function ({actions, observations} -> bool) that indicates that skill execution is done
    :param reset_start_rollout: whether to reset the env when calling this function
    :param keep_rendered_rgbs: whether to keep a list of all rgb_arrays (for future video making)
    :param animated: whether to render env after each step
    :param speedup: speedup factor for animation
    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    terminated = []
    rendered_rgbs = []
    if reset_start_rollout:
        env.reset()  # otherwise it will never advance!!
    o = env.unwrapped.get_current_obs()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
        rendered_rgbs.append(env.render(mode='rgb_array'))
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        # natural termination
        if d:
            terminated.append(1)
            break
        terminated.append(0)
        # skill decides to terminate
        if skill_stopping_func and skill_stopping_func(actions, observations):
            break

        o = next_o
        if keep_rendered_rgbs:  # will return a new entry to the path dict with all rendered images
            rendered_rgbs.append(env.render(mode='rgb_array'))
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    # This is off as in the case of being an inner rollout, it will close the outer renderer!
    # if animated:
    #     env.render(close=True)

    path_dict = dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),  # here it concatenates all lower-level paths!
        # termination indicates if the rollout was terminated or if we simply reached the limit of steps: important
        # when BOTH happened at the same time, to still be able to know it was the done (for hierarchized envs)
        terminated=tensor_utils.stack_tensor_list(terminated),
    )
    if keep_rendered_rgbs:
        path_dict['rendered_rgbs'] = tensor_utils.stack_tensor_list(rendered_rgbs)

    return path_dict
