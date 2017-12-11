from rllab.envs.grid_world_env import GridWorldEnv
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



class GridWorldRenderEnv(GridWorldEnv):
    def render(self):
        x = self.state // self.n_col
        y = self.state % self.n_col
        print('Current position: X = {:2}, Y = {:2}'.format(x, y))



class GridWorldObsEnv(GridWorldEnv):
    
    def __init__(self, desc='4x4', obs_dist=2):
        Serializable.quick_init(self, locals())
        super().__init__(desc)
        self.obs_dist = obs_dist
        self.obs_wide = self.obs_dist * 2 + 1
        desc_nums = np.zeros(self.desc.shape)
        desc_nums[self.desc == 'W'] = 1
        desc_nums[self.desc == 'H'] = 1
        pad_shape = np.array(self.desc.shape) + 2 * self.obs_dist
        self.desc_pad = np.ones(pad_shape)
        self.desc_pad[obs_dist:-obs_dist, obs_dist:-obs_dist] = desc_nums
        
        self.paths_plot_num = 0
        ## Live plotting
        # plt.ion()
    
    def state_to_observation(self, state):
        x = state // self.n_col
        y = state % self.n_col
        obs = np.copy(self.desc_pad[x : x+self.obs_wide , y : y+self.obs_wide])
        return obs
    
    def state_to_plt_xy(self, state):
        r = self.n_row - (state // self.n_col) - 1
        c = state % self.n_col
        return (c, r)

    def reset(self):
        state = super().reset()
        return self.state_to_observation(state)

    def step(self, action):
        state, reward, done, *_ = super().step(action)
        obs = self.state_to_observation(state)
        return Step(observation=obs, reward=reward, done=done, pos=self.state_to_plt_xy(state))

    @property
    def observation_space(self):
        return Box(low=0., high=1., shape=(self.obs_wide, self.obs_wide))
    
    @overrides
    def log_diagnostics(self, paths):
        # paths = [{ 'agent_infos': { 'prob': <2D numpy array> },
        #            'observations': <2D numpy array: T x |O|>,
        #            'actions': <2D numpy array: T x |A|>,
        #            'rewards': <1D numpy array>,
        #            'advantages': <1D numpy array>,
        #            'returns': <1D numpy array>,
        #            'env_infos': { <whatever extra step() returned>: numpy array }
        #          }, ...]
        
        plt.cla()
        plt.clf()
        x_grid = np.arange(self.n_col + 1) - 0.5
        y_grid = np.arange(self.n_row + 1) - 0.5
        plt.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='g', lw=1, alpha=0.8)
        plt.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='g', lw=1, alpha=0.8)
        plt.xlim(-0.5, self.n_col - 0.5)
        plt.ylim(-0.5, self.n_row - 0.5)
        plt.tight_layout()
            
        for path in paths:
            pos = path['env_infos']['pos'].T
            pos = np.c_[ np.array(self.state_to_plt_xy(self.start_state)), pos ]
            pos = pos + np.random.normal(size=pos.shape, scale=0.1)
            plt.plot(pos[0], pos[1], ls='-', c='midnightblue', alpha=0.05)
        
        ## Live plotting
        # plt.gcf().canvas.draw()
        # plt.waitforbuttonpress(timeout=0.001)
        
        # Save paths figure
        log_dir = logger.get_snapshot_dir()
        self.paths_plot_num += 1
        plt.savefig(os.path.join(log_dir, 'visitation{:0>3d}.png'.format(self.paths_plot_num)))
