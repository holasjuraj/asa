import numpy as np

from gym.spaces import Box, Discrete
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides

from sandbox.asa.envs import AsaEnv


MAPS = {
    "chain": ["GFFFFFFFFFFFFFSFFFFFFFFFFFFFG"],
    "4x4_safe": [
        "SFFF",
        "FWFW",
        "FFFW",
        "WFFG"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "8x8_move_goal": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFF"
    ],
}   # yapf: disable


class GridWorldEnv(AsaEnv, Serializable):
    """
    TODO add class description

    Maps legend:
        'S' : starting point
        'F' / '.' / ' ': free space
        'W' / 'x' / '#': wall
        'H' / 'O': hole (terminates episode)
        'G' : goal
    """
    MAP = [
            "S.......",
            "........",
            "...H....",
            ".....H..",
            "...H....",
            ".HH...H.",
            ".H..H.H.",
            "...H...G"
          ]
    STEP_PENALTY = 0  # DEBUG changed from 0.05


    # noinspection PyMissingConstructor
    def __init__(self):
        # Normalize char map
        m = np.array([list(row.upper()) for row in self.MAP])
        m[np.logical_or(m == '.', m == ' ')] = 'F'
        m[np.logical_or(m == 'X', m == '#')] = 'W'
        m[m == 'O'] = 'H'
        self.map = m

        # Set state
        self.agent_pos = None
        self.moving_goal = 'G' not in self.map
        self.n_row, self.n_col = self.map.shape

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())


    @property
    def action_space(self):
        """
        Up / right / down / left
        """
        return Discrete(4)

    @property
    def observation_space(self):
        """
        Position of agent ++ position of goal
        """
        return Box(low=np.array([0, 0, 0, 0]),
                   high=np.array([self.n_col, self.n_row, self.n_col, self.n_row]) - 1,
                   dtype=np.float32)


    @overrides
    def get_current_obs(self):
        """
        Position of agent ++ position of goal
        """
        r, c = self.agent_pos
        (goal_r,), (goal_c,) = np.nonzero(self.map == 'G')
        return np.array([r, c, goal_r, goal_c])


    def reset(self):
        """
        Initialize the agent.
        If self.moving_goal: randomly choose start point and goal.
        """
        if self.moving_goal:
            self.map[self.map == 'S'] = 'F'
            self.map[self.map == 'G'] = 'F'
            while True:
                r, c = np.random.randint(self.n_row), np.random.randint(self.n_col)
                if self.map[r, c] == 'F':
                    self.map[r, c] = 'S'
                    break
            while True:
                r, c = np.random.randint(self.n_row), np.random.randint(self.n_col)
                if self.map[r, c] == 'F':
                    self.map[r, c] = 'G'
                    break
        (start_r,), (start_c,) = np.nonzero(self.map == 'S')
        self.agent_pos = np.array([start_r, start_c])
        return self.get_current_obs()


    @overrides
    def reset_to_state(self, start_obs, **kwargs):
        """
        Choose state that matches given observation.
        """
        # TODO
        return self.reset()


    def step(self, action):
        """
        Action map = {0: up, 1: right, 2: down, 3: left}
        :param action: scalar encoding the action
        """
        # Set new state
        moves = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        next_pos = np.clip(self.agent_pos + moves[action],
                           a_min=[0, 0],
                           a_max=[self.n_row - 1, self.n_col - 1])
        next_state_type = self.map[next_pos[0], next_pos[1]]
        if next_state_type != 'W':
            self.agent_pos = next_pos
        else:
            next_state_type = 'F'  # agent stays on free tile

        # Determine reward and termination
        if next_state_type == 'H':
            done = True
            reward = -1
        elif next_state_type in ['F', 'S']:
            done = False
            reward = -self.STEP_PENALTY
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError

        # Return observation and others
        obs = self.get_current_obs()
        return Step(observation=obs,
                    reward=reward,
                    done=done)



    @overrides
    def render(self, mode='human'):
        pass


    @overrides
    def log_diagnostics(self, paths):
        pass
