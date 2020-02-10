import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from sandbox.asa.envs.asa_env import AsaEnv
from gym.spaces import Box
from garage.envs.base import Step

from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
from garage.misc import logger



class MinibotEnv(AsaEnv, Serializable):
    """
    Simulated two-wheeled robot in an environment with obstacles.

    Robot`s max. travel distance in one timestep is 0.2 (if both motors are set to 100% power).
    Robot`s wheels are on its sides, and robot is 0.764 wide (2.4/pi), i.e. robot can rotate 90
    degrees in 3 timesteps (if motors are at +100% and -100%).
    Robot always starts facing north (however, agent itself does not have a notion of its orientation).

    Robot is equipped with a 'radar' which scans presence of obstacles within its proximity.
    Radar output is a square bit matrix: presence/absence of obstacle in grid points
    [agent`s X position +- N*radar_resolution ; agent`s Y position +- M*radar_resolution]
    where 0 <= N,M <= radar_range.

    Agent`s actions are optionally discretized from {<-1,-0.33> , <-0.33,0.33> , <0.33,1>} to [-1, 0, 1],
    to ensure full motor output (bang-bang policy).


    Maps legend:
        'S' : starting point
        'F' / '.' / ' ': free space
        'W' / 'x' / '#': wall
        'H' / 'O': hole (terminates episode)
        'G' : goal
    """

    all_maps = [
        ["........",
         "........",
         "........",
         "........",
         "...G....",
         "..S##...",
         "...##...",
         "...##..."
        ],
        ["........",
         "........",
         "........",
         "........",
         "........",
         "..S##G..",
         "...##...",
         "...##..."
        ],
        ["........",
         "........",
         "........",
         "........",
         "........",
         "...##...",
         "...##...",
         "..S##G.."
        ],
        ["........",
         "........",
         "...##...",
         "...##...",
         "...##...",
         "...##...",
         "..S##G..",
         "...##..."
        ],
        ["........",
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
    EPSILON = 0.0001
    metadata = {'render.modes': ['rgb_array']}


    # noinspection PyMissingConstructor
    def __init__(self,
                 radar_range=2,
                 radar_resolution=1,
                 discretized=True,
                 use_maps='all',
                 states_cache=None):
        """
        :param radar_range: how many measurements does 'radar' make to each of 4 sides (and combinations)
        :param radar_resolution: distance between two measurements of agent`s 'radar'
        :param discretized: discretized actions from {<-1,-0.33> , <-0.33,0.33> , <0.33,1>} to [-1, 0, 1]
        :param use_maps: which maps to use, list of indexes or 'all'
        :param states_cache: pre-populated cache to use (observation -> set of states)
        """
        Serializable.quick_init(self, locals())

        self.radar_range = radar_range
        self.radar_resolution = radar_resolution
        self.discretized = discretized
        if states_cache is None:
            self.states_cache = dict()
        else:
            self.states_cache = states_cache
        self.agent_width = 2.4/np.pi
        self.max_action_distance = 0.2
        self.do_render_init = True
        self.render_prev_pos = np.zeros(2)
        self.do_caching = True

        self.current_map_idx = None
        self.agent_pos = None
        self.agent_ori = None

        # Maps initialization
        if use_maps == 'all':
            raw_maps = self.all_maps
        else:
            # noinspection PyTypeChecker
            raw_maps = [self.all_maps[i] for i in use_maps]
        self.maps = []
        self.bit_maps = []
        for i in range(len(raw_maps)):
            # Normalize char map
            m = np.array([list(row.upper()) for row in raw_maps[i]])
            m[np.logical_or(m == '.', m == ' ')] = 'F'
            m[np.logical_or(m == 'X', m == '#')] = 'W'
            m[m == 'O'] = 'H'
            self.maps.append(m)
            # Make bit map
            bm = np.zeros(m.shape)
            bm[np.logical_or(m == 'W', m == 'H')] = 1
            self.bit_maps.append(bm)


    @property
    def action_space(self):
        """
        Power on left/right motor
        """
        return Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    @property
    def observation_space(self):
        """
        0 = free space, 1 = wall/hole
        """
        obs_wide = self.radar_range * 2 + 1
        return Box(low=0, high=1, shape=(obs_wide, obs_wide), dtype=np.float32)


    @overrides
    def get_current_obs(self):
        """
        Get what agent can see (up to radar_range distance), rotated according
        to agent`s orientation.
        """
        bound = self.radar_range * self.radar_resolution
        ls = np.linspace(-bound, bound, 2 * self.radar_range + 1)
        xx, yy = np.meshgrid(ls, ls)
        points = np.array([xx.flatten(), yy.flatten()])
        # Transform points from agent`s coordinates to world coordinates
        r = self._rotation_matrix(self.agent_ori)
        points = self.agent_pos + (r @ points).T
        # Fill in observation vector
        obs = np.zeros(points.shape[0])
        for i, point in enumerate(points):
            obs[i] = self._tile_type_at_pos(point, bitmap=True)
        return obs


    @overrides
    def reset(self):
        """
        Choose random map for this rollout, initialize agent facing north.
        """
        self.do_render_init = True
        self.current_map_idx = np.random.choice(len(self.maps))
        m = self.maps[self.current_map_idx]
        (start_r,), (start_c,) = np.nonzero(m == 'S')
        self.agent_pos = self._rc_to_xy([start_r, start_c])
        self.agent_ori = 0
        return self.get_current_obs()


    @overrides
    def reset_to_state(self, start_obs, **kwargs):
        """
        Choose state that matches given observation.
        """
        self.do_render_init = True
        k = tuple(np.array(start_obs, dtype='int8'))
        states = list(self.states_cache[k])
        m_idx, pos, ori = states[np.random.randint(len(states))]
        pos = np.array(pos)
        self.current_map_idx = m_idx
        self.agent_pos = pos
        self.agent_ori = ori
        return self.get_current_obs()


    @overrides
    def step(self, action):
        """
        :param action: power on left & right motor
        :type action: numpy.ndarray
        """
        if self.discretized:
            # discretization from {<-1,-0.33> , <-0.33,0.33> , <0.33,1>} to [-1, 0, 1]
            action = np.clip(np.round(action * 1.5), a_min=-1, a_max=1)
        # Set new state
        next_pos, next_ori = self._get_new_position(action)
        self.agent_pos = next_pos
        self.agent_ori = next_ori
        obs = self.get_current_obs()
        # Determine reward and termination
        next_state_type = self._tile_type_at_pos(next_pos)
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

        # Cache observation
        if self.do_caching:
            k = tuple(np.array(obs, dtype='int8'))
            v = (self.current_map_idx, tuple(self.agent_pos), self.agent_ori)
            if k not in self.states_cache:
                self.states_cache[k] = {v}
            else:
                self.states_cache[k].add(v)

        # Return observation and others
        return Step(observation=obs, reward=reward, done=done,
                    agent_pos=self.agent_pos, agent_ori=self.agent_ori,
                    map=self.current_map_idx)


    @overrides
    def render(self, mode='rgb_array'):
        if self.do_render_init:
            self.do_render_init = False
            self.render_prev_pos = self.agent_pos
            # Initialize plotting
            ax = plt.gca()
            plt.cla()
            # Plot cells grid
            m = self._get_current_map()
            rows, cols = m.shape
            ax.set_xlim(-0.5, cols - 0.5)
            ax.set_ylim(-0.5, rows - 0.5)
            # Grid
            x_grid = np.arange(rows + 1) - 0.5
            y_grid = np.arange(cols + 1) - 0.5
            ax.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='k', lw=1, alpha=0.8)
            ax.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='k', lw=1, alpha=0.8)
            # Start, goal, walls and holes
            start = self._rc_to_xy(  np.argwhere(m == 'S').T  , rows)
            goal  = self._rc_to_xy(  np.argwhere(m == 'G').T  , rows)
            holes = self._rc_to_xy(  np.argwhere(m == 'H').T  , rows)
            walls = self._rc_to_xy(  np.argwhere(m == 'W').T  , rows)
            ax.scatter(*start, c='r', marker='o', s=50 )
            ax.scatter(*goal,  c='r', marker='x', s=50 )
            ax.scatter(*holes, c='k', marker='v', s=100)
            ax.add_collection(PatchCollection([Rectangle(xy-0.5, 1, 1) for xy in walls.T], color='navy'))
            # Agent`s start
            ax.scatter(*self.agent_pos, marker='x', s=50, c='g')
        # Plot last move
        move = np.array([self.render_prev_pos, self.agent_pos]).T
        plt.scatter(*move, marker='.', s=6, c='k')
        plt.plot(*move, '-', c=self.map_colors[self.current_map_idx % len(self.map_colors)])
        self.render_prev_pos = self.agent_pos

        # Choose output method
        mode = 'rgb-array'  # DEBUG
        if mode == 'human':
            # plt.show(block=False)
            raise NotImplementedError
        elif mode == 'rgb-array':
            pass
            # dir = logger.get_snapshot_dir()
            # if dir is None:
            #     dir = '~/garage/data/local/asa/instant-run';
            # dir = os.path.expanduser(dir)
            # if not os.path.isdir(dir):
            #     os.makedirs(dir)
            # plt.savefig(os.path.join(dir, 'demo_run.png'))
        else:
            super(MinibotEnv, self).render(mode=mode)  # just raise an exception


    # noinspection PyMethodMayBeStatic
    def save_rendered_plot(self):
        plt.scatter(*self.agent_pos, marker='x', s=50, c='r')  # DEBUG to mark agent`s end position
        directory = logger.get_snapshot_dir()
        if directory is None:
            directory = '~/garage/data/local/asa/instant-run'
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        base = 'demo_run_'
        try:
            i = 1 + max([int(f[len(base):f.find('.')]) for f in os.listdir(directory) if f.startswith(base)])
        except ValueError:
            i = 0
        plt.savefig(os.path.join(directory, '{}{}.png'.format(base, i)))


    def _get_new_position(self, action):
        """
        Using current state and given action, return new state (position, orientation) considering walls.
        Does not change the agent`s position
        :return: tuple(position, orientation)
        """
        pos0 = self.agent_pos
        ori0 = self.agent_ori
        move_vector, ori_change = self._get_raw_move(action)
        pos1 = pos0 + move_vector
        ori1 = ori0 + ori_change

        # Collision detection
        t0 = np.round(pos0)  # starting tile
        t1 = np.round(pos1)  # ending tile
        if self._tile_type_at_pos(pos1) != 'W':  # no collision (t1=='W' also implies t1 != t0)
            return pos1, ori1

        dtx = t0[0] != t1[0]  # was tile changed in x-direction
        dty = t0[1] != t1[1]  # was tile changed in y-direction
        mid_x, mid_y = (t0 + t1) / 2
        near_wall_x = mid_x - np.sign(move_vector[0]) * 0.01
        near_wall_y = mid_y - np.sign(move_vector[1]) * 0.01
        if dtx and not dty:  # bumped into wall E/W
            pos1[0] = near_wall_x
            return pos1, ori1
        if dty and not dtx:  # bumped into wall N/S
            pos1[1] = near_wall_y
            return pos1, ori1

        # now we know: t1=='W', dtx, dty ... i.e. we moved diagonally, traversing another tile either on E/W xor on N/S
        t_ew_wall = self._tile_type_at_pos((t1[0], t0[1])) == 'W'
        t_ns_wall = self._tile_type_at_pos((t0[0], t1[1])) == 'W'
        if not t_ew_wall  and  not t_ns_wall:
            tx = (mid_x - pos0[0]) / (pos1[0] - pos0[0])
            ty = (mid_y - pos0[1]) / (pos1[1] - pos0[1])
            if tx < ty:  # traveled through empty tile on E/W, bumped into wall on N/S
                pos1[1] = near_wall_y
            else:            # traveled through empty tile on N/S, bumped into wall on E/W
                pos1[0] = near_wall_x
            return pos1, ori1
        if t_ew_wall:  # bumped into wall on E/W
            pos1[0] = near_wall_x
        if t_ns_wall:  # bumped into wall on N/S
            pos1[1] = near_wall_y
        # combination of last two: bumped into corner
        return pos1, ori1


    def _get_raw_move(self, action):
        """
        Computes:
        1) a vector: in which direction the agent should move (in world coordinates)
        2) orientation change
        Does not handle collisions, nor does it actually changes agent`s position.
        """
        al, ar = action * self.max_action_distance  # scale motor power <-1,1> to actual distance <-0.2,0.2>
        w = self.agent_width

        if np.abs(al - ar) < self.EPSILON:
            # al == ar -> Agent moves in straight line
            relative_move_vector = np.array([0, al])
            ori_change = 0
        elif np.abs(al + ar) < self.EPSILON:
            # al == -ar -> Agent rotates in place
            relative_move_vector = np.array([0, 0])
            ori_change = ar * 2 / w
        else:
            # Agent moves and rotates at the same time
            r = (w * (ar + al)) / (2 * (ar - al))
            alpha = (ar + al) / (2 * r)
            me_pos = np.array([0, 0])  # agent`s position in his frame of reference
            rot_center = np.array([-r, 0])
            me_pos = me_pos - rot_center                      # 1) move to rotation center
            me_pos = self._rotation_matrix(alpha) @ me_pos    # 2) rotate
            me_pos = me_pos + rot_center                      # 3) move back
            # Vector me_pos now represents in which direction the agent should move !!! in his frame of reference !!!
            relative_move_vector = me_pos
            ori_change = alpha
        absolute_move_vector = self._rotation_matrix(self.agent_ori) @ relative_move_vector
        return absolute_move_vector, ori_change


    @staticmethod
    def _rotation_matrix(alpha):
        """
        2-D rotation matrix for column-vectors (i.e. usage: x' = R @ x).
        """
        sin = np.sin(alpha)
        cos = np.cos(alpha)
        return np.array([[cos, -sin], [sin, cos]])


    def _tile_type_at_pos(self, position, bitmap=False):
        """
        Return type of a tile at given X,Y position.
        """
        m = self._get_current_map(bitmap)
        rows, cols = m.shape
        x, y = np.round(position)
        r, c = self._xy_to_rc([x, y])
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return 1 if bitmap else 'W'
        return m[r, c]


    def _rc_to_xy(self, pos, rows=None):
        """
        Get position as [X,Y], instead of [row, column].
        :param pos: (r,c) position
        :param rows: number of rows in a map, or None (current map is used)
        """
        r, c = pos
        if rows is None:
            rows, _ = self._get_current_map().shape
        return np.array([c, rows - r - 1])

    def _xy_to_rc(self, pos, rows=None):
        """
        Get position as [row, column], instead of [X,Y]. Original [X,Y] position will be rounded to nearest tile.
        :param pos: (x,y) position
        :param rows: number of rows in a map, or None (current map is used)
        """
        x, y = np.round(pos)
        if rows is None:
            rows, _ = self._get_current_map().shape
        return np.array([rows - y - 1, x], dtype='int32')


    def _get_current_map(self, bitmap=False):
        """
        Return current map, or bitmap (for observations).
        """
        if bitmap:
            return self.bit_maps[self.current_map_idx]
        return self.maps[self.current_map_idx]
