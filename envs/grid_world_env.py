import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
from gym.spaces import Box

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


class GridWorldEnv(gym.Env, Serializable):
    """
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal


    """

    def __init__(self, desc='4x4'):
        self.orig_desc = desc
        plan = desc
        if isinstance(desc, str):
            plan = MAPS[desc]
        plan = np.array(list(map(list, plan)))
        plan[plan == '.'] = 'F'
        plan[plan == 'o'] = 'H'
        plan[plan == 'x'] = 'W'
        self.desc = plan
        self.n_row, self.n_col = plan.shape
        (start_x, ), (start_y, ) = np.nonzero(plan == 'S')
        self.start_state = start_x * self.n_col + start_y
        self.state = None
        self.domain_fig = None

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def reset(self):
        if self.orig_desc == '8x8_move_goal':
            self.desc[self.desc == 'S'] = 'F'
            self.desc[self.desc == 'G'] = 'F'
            while True:
                x, y = np.random.randint(self.n_col), np.random.randint(self.n_row)
                if self.desc[x, y] == 'F':
                    self.desc[x, y] = 'S'
                    self.start_state = x * self.n_col + y
                    break
            while True:
                x, y = np.random.randint(self.n_col), np.random.randint(self.n_row)
                if self.desc[x, y] == 'F':
                    self.desc[x, y] = 'G'
                    break
        self.state = self.start_state
        start_x = self.state // self.n_col
        start_y = self.state % self.n_col
        (goal_x,), (goal_y,) = np.nonzero(self.desc == 'G')
        return np.array([start_x, start_y, goal_x, goal_y])

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a
        helper method for debugging and testing purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(left=0, down=1, right=2, up=3)[d]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        possible_next_states = self.get_possible_next_states(
            self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_x = next_state // self.n_col
        next_y = next_state % self.n_col

        next_state_type = self.desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError
        self.state = next_state
        (goal_x,), (goal_y,) = np.nonzero(self.desc == 'G')
        return Step(observation=np.array([next_x, next_y, goal_x, goal_y]),
                    reward=reward,
                    done=done)

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and
        their probabilities. Only next states with nonzero probabilities will
        be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """
        # assert self.observation_space.contains(state)
        # assert self.action_space.contains(action)

        x = state // self.n_col
        y = state % self.n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(coords + increments[action], [0, 0],
                              [self.n_row - 1, self.n_col - 1])
        next_state = next_coords[0] * self.n_col + next_coords[1]
        state_type = self.desc[x, y]
        next_state_type = self.desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]

    @property
    def action_space(self):
        return gym.spaces.Discrete(4)

    @property
    def observation_space(self):
        # return gym.spaces.Discrete(self.n_row * self.n_col)
        return Box(low=np.array([0, 0, 0, 0]),
                   high=np.array([self.n_col, self.n_row, self.n_col, self.n_row]) - 1,
                   dtype=np.float32)

    @overrides
    def render(self, mode='human'):
        pass

    @overrides
    def log_diagnostics(self, paths):
        pass
