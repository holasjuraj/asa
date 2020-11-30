import numpy as np
from garage.core import Serializable
from garage.policies import Policy
from sandbox.asa.envs import GridWorldGathererEnv


class GridworldTargetPolicy(Policy, Serializable):
    """
    Policy with fixed behaviour of navigating agent to the target tile. Agent
    takes the shortest route, but only does very primitive obstacle avoidance.
    """

    def __init__(self, env_spec, target):
        """
        :param target: [row, column] of target tile
        """
        self.target = np.array(target)
        Serializable.quick_init(self, locals())
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        moves = {(-1,  0): 0,
                 ( 0,  1): 1,
                 ( 1,  0): 2,
                 ( 0, -1): 3,
                 ( 0,  0): 0  # when agent is already on target tile, go up
                }
        pos = observation[:2]

        # Get primary & secondary moves
        delta = self.target - pos
        abs_d = np.abs(delta)
        move_r, move_c = np.diag( delta // abs_d )
        move_1, move_2 = (move_r, move_c) if abs_d[0] >= abs_d [1] else (move_c, move_r)

        # If primary move goes out of map / into wall, use secondary move
        next_pos = pos + move_1
        m = np.asarray(GridWorldGathererEnv.MAP)
        if np.min(next_pos) < 0  \
           or  np.any(next_pos >= m.shape)  \
           or  m[next_pos[0], next_pos[1]] == 'W':
            move_1 = move_2

        return moves[tuple(move_1)], dict()

    def get_params_internal(self, **tags):
        return [self.target]
