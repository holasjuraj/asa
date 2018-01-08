from rllab.envs.base import Env
from rllab.spaces import Discrete, Box
from rllab.envs.base import Step

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
import os

import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt



class GridMazeEnv(Env, Serializable):
    '''
    'S' : starting point
    'F' / '.' / ' ': free space
    'W' / 'x' / '#': wall
    'H' / 'O': hole (terminates episode)
    'G' : goal
    '''
    maps = [
        ["........",
         "........",
         "........",
         "...##...",
         "...##...",
         "...##...",
         "..s##g..",
         "...##..."
        ],
        ["........",
         "........",
         "........",
         "........",
         "...##...",
         "...##...",
         "...##...",
         "..S##G.."
        ]
    ]
    map_colors = ['maroon', 'midnightblue', 'darkgreen', 'darkgoldenrod']

    def __init__(self, obs_dist=2, plot=None):
        '''
        :param obs_dist: how far agent sees, Manhattan distance from agent, default=2
        :param plot: {'save':<path>, 'live':<bool>, 'alpha':<0..1>}
        '''
        Serializable.quick_init(self, locals())
        
        self.obs_dist = obs_dist
        self.obs_wide = self.obs_dist * 2 + 1
        self.current_map_idx = None
        self.agent_pos = None
        self.agent_ori = None
        
        # Maps initialization
        self.bit_maps = []
        for i in range(len(self.maps)):
            # Normalize char map
            m = np.array([list(row.upper()) for row in self.maps[i]])
            m[np.logical_or(m == '.', m == ' ')] = 'F'
            m[np.logical_or(m == 'X', m == '#')] = 'W'
            m[m == 'O'] = 'H'
            self.maps[i] = m
            # Make bit map
            bm_inner = np.zeros(m.shape)
            bm_inner[np.logical_or(m == 'W', m == 'H')] = 1
            pad_shape = np.array(m.shape) + 2 * self.obs_dist
            bm_padded = np.ones(pad_shape)
            bm_padded[obs_dist:-obs_dist, obs_dist:-obs_dist] = bm_inner
            self.bit_maps.append(bm_padded)
            
        # Plotting
        self.paths_plot_num = 0
        if plot is None:
            self.plot_opts = {}
        else:
            self.plot_opts = plot
        if self.plot_opts.get('live', False):
            plt.ion()


    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Box(low=0., high=1., shape=(self.obs_wide, self.obs_wide))


    def reset(self):
        self.current_map_idx = np.random.choice(len(self.maps))
        m = self.maps[self.current_map_idx]
        (start_r,), (start_c,) = np.nonzero(m == 'S')
        self.agent_pos = np.array([start_r, start_c])
        self.agent_ori = 0
        return self.get_observation()
    

    def step(self, action):
        '''
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
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
        Using current state and given action, return a list of possible next states and their probabilities.
        Only next states with nonzero probabilities will be returned.
        :param action: action
        :return: a list of pairs (s', p(s'|s,a)), where s` is tuple (position, orientation)
        '''
        ### TODO: add agent`s orientation
        r, c = self.agent_pos
        m = self.get_current_map()
        rows, cols = m.shape
        
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_pos = np.clip(
            self.agent_pos + increments[action],
            [0, 0],
            [rows - 1, cols - 1]
        )
        
        state_type = m[r, c]
        next_state_type = m[next_pos[0], next_pos[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [((self.agent_pos, 0), 1.)]
        else:
            return [((next_pos, 0), 1.)]
    
    
    def get_observation(self):
        ### TODO: rotate observation according to agent`s orientation
        bm = self.get_current_map(bitmap=True)
        r, c = self.agent_pos
        obs = np.copy(bm[r : r+self.obs_wide , c : c+self.obs_wide])
        return obs
 
    
    def get_pos_as_xy(self, pos=None, rows=None):
        if pos is None:
            pos = self.agent_pos
        r, c = pos
        if rows is None:
            rows, _ = self.get_current_map().shape
        return (c, rows - r - 1)
    
    
    def get_current_map(self, bitmap=False):
        if bitmap:
            return self.bit_maps[self.current_map_idx]
        return self.maps[self.current_map_idx]
    
    
    @overrides
    def log_diagnostics(self, paths):
        '''
        Plot all paths in current batch.
        '''
        if len(self.plot_opts) == 0:
            return
        plt.cla()
        plt.clf()
        # Plot cells grid
        m = self.get_current_map()
        rows, cols = m.shape
        x_grid = np.arange(rows + 1) - 0.5
        y_grid = np.arange(cols + 1) - 0.5
        plt.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='g', lw=1, alpha=0.8)
        plt.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='g', lw=1, alpha=0.8)
        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.tight_layout()
        # Plot paths
        alpha = self.plot_opts.get('alpha', 0.1)
        for path in paths:
            # Starting position
            map_idx = path['env_infos']['map'][0]
            m = self.maps[map_idx]
            (start_r,), (start_c,) = np.nonzero(m == 'S')
            start_pos_rc = np.array([start_r, start_c])
            start_pos_xy = self.get_pos_as_xy(pos=start_pos_rc, rows=m.shape[0])
            # All others
            pos = path['env_infos']['pos_xy'].T
            pos = np.c_[start_pos_xy, pos]
            pos = pos + np.random.normal(size=pos.shape, scale=0.1)
            c = self.map_colors[map_idx % len(self.map_colors)]
            plt.plot(pos[0], pos[1], ls='-', c=c, alpha=alpha)
        # Save paths figure
        dir = self.plot_opts.get('save', False)
        if dir:
            if isinstance(dir, str):
                dir = os.path.expanduser(dir)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
            else:
                dir = logger.get_snapshot_dir()
            plt.savefig(os.path.join(dir, 'visitation{:0>3d}.png'.format(self.paths_plot_num)))
            self.paths_plot_num += 1
        # Live plotting
        if self.plot_opts.get('live', False):
            plt.gcf().canvas.draw()
            plt.waitforbuttonpress(timeout=0.001)

