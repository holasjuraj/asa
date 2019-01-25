import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

# from garage.envs.base import GarageEnv
from gym import Env
from gym.spaces import Discrete, Box
from garage.envs.base import Step

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger
import os
from sandbox.asa.tools.path_trie import PathTrie




class GridMazeEnv(Env, Serializable):
    '''
    Maps legend:
        'S' : starting point
        'F' / '.' / ' ': free space
        'W' / 'x' / '#': wall
        'H' / 'O': hole (terminates episode)
        'G' : goal
    
    Action map:
        0: turn left
        1: step
        2: turn right
    
    Orientations map:
        0: up
        1: right
        2: down
        3: left
    '''
    
    all_maps = [
        ["........", # min = 11 actions
         "........",
         "........",
         "........",
         "........",
         "...##...",
         "...##...",
         "..S##G.."
        ],
        ["........", # min = 13 actions
         "........",
         "...##...",
         "...##...",
         "...##...",
         "...##...",
         "..S##G..",
         "...##..."
        ],
        ["........", # min = 21 actions
         "........",
         "...###..",
         "..S###..",
         "...###..",
         "######..",
         "..G###..",
         "........"
        ]
    ]
    map_colors = ['maroon', 'midnightblue', 'darkgreen', 'darkgoldenrod']

    def __init__(self, obs_dist=2, use_maps='all'):
        '''
        :param obs_dist: how far agent sees, Manhattan distance from agent, default=2
        :param use_maps: which maps to use, list of indexes or 'all'
        '''
        Serializable.quick_init(self, locals())
        
        self.obs_dist = obs_dist
        self.obs_wide = self.obs_dist * 2 + 1
        self.current_map_idx = None
        self.agent_pos = None
        self.agent_ori = None
        
        # Maps initialization
        if use_maps == 'all':
            maps = self.all_maps
        else:
            maps = [self.all_maps[i] for i in use_maps]
        self.maps = []
        self.bit_maps = []
        for i in range(len(maps)):
            # Normalize char map
            m = np.array([list(row.upper()) for row in maps[i]])
            m[np.logical_or(m == '.', m == ' ')] = 'F'
            m[np.logical_or(m == 'X', m == '#')] = 'W'
            m[m == 'O'] = 'H'
            self.maps.append(m)
            # Make bit map
            bm_inner = np.zeros(m.shape)
            bm_inner[np.logical_or(m == 'W', m == 'H')] = 1
            pad_shape = np.array(m.shape) + 2 * self.obs_dist
            bm_padded = np.ones(pad_shape)
            bm_padded[obs_dist:-obs_dist, obs_dist:-obs_dist] = bm_inner
            self.bit_maps.append(bm_padded)


    @property
    def action_space(self):
        '''
        turn left / step (no turn right)
        '''
        return Discrete(2)

    @property
    def observation_space(self):
        '''
        0 = free space, 1 = wall/hole
        '''
        return Box(low=0, high=1, shape=(self.obs_wide, self.obs_wide), dtype=np.float32)


    def reset(self):
        '''
        Choose random map for this rollout, init agent facing north.
        '''
        self.current_map_idx = np.random.choice(len(self.maps))
        m = self.maps[self.current_map_idx]
        (start_r,), (start_c,) = np.nonzero(m == 'S')
        self.agent_pos = np.array([start_r, start_c])
        self.agent_ori = 0
        return self.get_observation()
    

    def step(self, action):
        '''
        Action map:
            0: turn left
            1: step
            2: turn right
        :param action: scalar encoding the action
        '''
        # Get next state possibilities
        possible_next_states = self.get_possible_next_states(action)
        # Sample next state from possibilities
        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_pos, next_ori = possible_next_states[next_state_idx][0]
        # Set new state
        self.agent_pos = next_pos
        self.agent_ori = next_ori
        m = self.get_current_map()
        next_r, next_c = next_pos
        # Determine reward and termination
        next_state_type = m[next_r, next_c]
        if next_state_type == 'H':
            done = True
            reward = -1
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError
        # Return observation (and others)
        obs = self.get_observation()
        return Step(observation=obs, reward=reward, done=done,
                    pos_xy=self.get_pos_as_xy(), map=self.current_map_idx)


    def get_possible_next_states(self, action):
        '''
        Using current state and given action, return a list of possible next
        states and their probabilities. Only next states with nonzero
        probabilities will be returned.
        Orientations map:
            0: up
            1: right
            2: down
            3: left
        :return: a list of pairs (s', p(s'|s,a)), where s` is tuple (position, orientation)
        '''
        r, c = self.agent_pos
        ori = self.agent_ori
        m = self.get_current_map()
        rows, cols = m.shape
        
        if action == 1:  # step forward
            deltas = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
            step_delta = deltas[ori]
            next_pos = np.clip(
                self.agent_pos + step_delta,
                [0, 0],
                [rows - 1, cols - 1]
            )
            next_ori = ori
        else:   # turn left/right
            next_pos = self.agent_pos
            turn = -1  if action == 0  else 1
            next_ori = (ori + turn + 4) % 4
        
        state_type = m[r, c]
        next_state_type = m[next_pos[0], next_pos[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [((self.agent_pos, next_ori), 1.)]
        else:
            return [((next_pos, next_ori), 1.)]
    
    
    def get_observation(self):
        '''
        Get what agent can see (up to obs_dist distance), rotated according
        to agent`s orientation.
        Orientations map:
            0: up
            1: right
            2: down
            3: left
        '''
        bm = self.get_current_map(bitmap=True)
        r, c = self.agent_pos
        obs = np.copy(
                np.rot90(
                    bm[r : r+self.obs_wide , c : c+self.obs_wide],
                    self.agent_ori
                    )
                )
        return obs
 
    
    def get_pos_as_xy(self, pos=None, rows=None):
        '''
        Get agent`s position as [X,Y], instead of [row, column].
        :param pos: (r,c) position of agent, or None (current position is used)
        :param rows: number of rows in a map, or None (current map is used)
        '''
        if pos is None:
            pos = self.agent_pos
        r, c = pos
        if rows is None:
            rows, _ = self.get_current_map().shape
        return np.array([c, rows - r - 1])
    
    
    def get_current_map(self, bitmap=False):
        '''
        Return current map, or bitmap (for observations).
        '''
        if bitmap:
            return self.bit_maps[self.current_map_idx]
        return self.maps[self.current_map_idx]
