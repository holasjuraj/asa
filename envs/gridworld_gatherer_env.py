import os
import numpy as np
import matplotlib
# matplotlib.use('qt5Agg')  # DEBUG
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import Rectangle

from gym.spaces import Box, Discrete
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
from garage.misc import logger

from sandbox.asa.envs import AsaEnv
from sandbox.asa.envs import SubpolicyPathInfo



class GridworldGathererEnv(AsaEnv, Serializable):
    """
    Coin gatherer in gridworld environment.
    Agent have to deliver all coins to goal positions. Agent can carry only one
    coin at a time. A coin is picked/dropped automatically when agent comes to
    coin tile/goal tile.
    Agent's position is initialized randomly in one of starting tiles. The
    rollout ends when all coins have been delivered.

    Actions: move up/right/down/left
    Observations: concatenation of:
        - agent's position on map (row, column)
        - whether or not agent holds a coin
        - which coins have been picked from the map (n-bit vector)

    Maps legend:
        'S' : starting point
        'F' / '.' / ' ': free space
        'W' / 'x' / '#': wall
        'H' / 'O': hole (terminates episode)
        'C' : coin
        'G' : goal
    """
    # MAP = [  # No coins, holes ahead
    #         "SSS.....",
    #         "SSS.....",
    #         "SSS#....",
    #         ".....#..",
    #         "...#....",
    #         ".#H...#.",
    #         ".H..#.#.",
    #         ".G.#.GGG"
    #       ]

    # MAP = [  # Single coin, medium
    #         "....#...",
    #         "SS..#...",
    #         "SS..#...",
    #         "SS..#..C",
    #         "SS......",
    #         "SS..#.##",
    #         "SS..#...",
    #         "....#G.."
    #       ]

    # MAP = [  # Multiple coins, small
    #         "#C#.",
    #         "#..#",
    #         "S..C",
    #         "#G..",
    #         "#...",
    #         "#..."
    #       ]

    # MAP = [  # Multiple coins, medium easier
    #         "....#...",
    #         "....#...",
    #         "SS..#C..",
    #         "SS.. ..C",
    #         "SS......",
    #         "SS..#...",
    #         "....#...",
    #         "....#..G"
    #       ]

    # MAP = [  # Multiple coins, medium harder
    #         "....#C.C",
    #         "SS..#...",
    #         "SS..#...",
    #         "SS..#..C",
    #         "SS......",
    #         "SS..#.##",
    #         "SS..#...",
    #         "....#..G"
    #       ]

    # MAP = [  # Test arena
    #         "#####################",
    #         "#...................#",
    #         "#...................#",
    #         "#....C..............#",
    #         "#.....##............#",
    #         "#...................#",
    #         "#...................#",
    #         "#.............#.....#",
    #         "#.............#..S..#",
    #         "#...................#",
    #         "#####################"
    #        ]

    MAP = [  # Test arena - small
            "###########################",
            "#....#....#.....#.........#",
            "#.C..#..C.#..C..#.SSSSSS..#",
            "#....#....#.....#.SSSSSS..#",
            "#....#....#.....#.SSSSSS..#",
            "##.#####.####.###.SSSSSS..#",
            "#.................SSSSSS..#",
            "#.................SSSSSS..#",
            "#.........................#",
            "#.........................#",
            "#.........................#",
            "#.........................#",
            "##.#####.####.####......###",
            "#..........#....#.........#",
            "#..........#....#.........#",
            "#.C.....C..#.C..#.........#",
            "#..........#....#GGGGGGGGG#",
            "#..........#....#GGGGGGGGG#",
            "###########################"
        ]

    # MAP = [  # Test arena
    #         "####################################################################",
    #         "#..........#.............#...............#.........................#",
    #         "#..........#.............#...............#.........................#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#....C.....#......C......#.......C.......#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#..........#.............#...............#....SSSSSSSSSSSSSSSSS....#",
    #         "#####..###########..############..########....SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#.............................................SSSSSSSSSSSSSSSSS....#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#..................................................................#",
    #         "#####..##########..##############..###########.................#####",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#....C............C.......#......C.......#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#.........................#",
    #         "#.........................#..............#GGGGGGGGGGGGGGGGGGGGGGGGG#",
    #         "#.........................#..............#GGGGGGGGGGGGGGGGGGGGGGGGG#",
    #         "#.........................#..............#GGGGGGGGGGGGGGGGGGGGGGGGG#",
    #         "####################################################################"
    #        ]

    STEP_PENALTY = 0


    # noinspection PyMissingConstructor
    def __init__(self, plot=None):
        """
        :param plot: which plots to generate. Only 'visitation' plot is supported now.
                {'visitation': <opts>}
                where opts = {'save': <directory or False>, 'live': <boolean> [, 'alpha': <0..1>][, 'noise': <0..1>]}
        """
        # Normalize char map
        m = np.array([list(row.upper()) for row in self.MAP])
        m[np.logical_or(m == '.', m == ' ')] = 'F'
        m[np.logical_or(m == 'X', m == '#')] = 'W'
        m[m == 'O'] = 'H'
        self.map = m
        self.n_row, self.n_col = self.map.shape

        # Set state
        self.coins_pos = np.argwhere(self.map == 'C')
        self.coins_num = self.coins_pos.shape[0]
        self.coins_picked = np.array([False] * self.coins_num)
        self.coin_holding = False
        self.agent_pos = None

        # Plotting
        self.visitation_plot_num = 0
        if (plot is None) or (plot is False):
            self.plot_opts = {}
        else:
            self.plot_opts = plot
        for plot_type in ['visitation']:
            if plot_type not in self.plot_opts  or  not isinstance(self.plot_opts[plot_type], dict):
                self.plot_opts[plot_type] = {}
        if any([plot_type_opts.get('live', False) for plot_type_opts in self.plot_opts.values()]):
            plt.ion()

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
        [Position of agent] + [coin holding] + [coins picked]
        - position of agent: [row, col]
        - coin holding: flag whether agent is holding a coin (1) or not (0)
        - coins picked: vector, for each coin 0 if it has been picked, 0 if not
        """
        high = np.concatenate((
            [self.n_col, self.n_row],
            [1],
            [1] * self.coins_num
        ))
        low = np.zeros_like(high)
        return Box(low, high, dtype=np.float32)


    @overrides
    def get_current_obs(self):
        """
        [Position of agent] + [coin holding] + [coins picked]
        - position of agent: [row, col]
        - coin holding: flag whether agent is holding a coin (1) or not (0)
        - coins picked: vector, for each coin 0 if it has been picked, 0 if not
        """
        return np.concatenate((
            self.agent_pos,
            [self.coin_holding],
            self.coins_picked
        ))


    def reset(self):
        """
        Initialize the agent positioned randomly in one of starting points.
        """
        starts = np.argwhere(self.map == 'S')
        n_starts = starts.shape[0]
        self.agent_pos = starts[np.random.randint(n_starts)]
        self.coins_picked = np.array([False] * self.coins_num)
        self.coin_holding = False
        return self.get_current_obs()


    @overrides
    def reset_to_state(self, start_obs, **kwargs):
        """
        Initialize the agent in a state that matches given observation.
        """
        obs = np.asarray(np.round(start_obs), dtype='int32')
        self.agent_pos = obs[:2]
        self.coin_holding = obs[2] > 0.5
        self.coins_picked = obs[3:] > 0.5
        return self.get_current_obs()


    def step(self, action):
        """
        Action map = {0: up, 1: right, 2: down, 3: left}
        :param action: scalar encoding the action
        """
        prev_pos_xy = self._get_pos_as_xy()
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
        reward = -self.STEP_PENALTY  # default reward
        done = False                 # default done state

        if next_state_type == 'H':  # hole
            # fall into abyss
            done = True
            reward = -1

        elif next_state_type == 'C':  # coin
            coin_idx = np.where(np.all(self.coins_pos == self.agent_pos, axis=1))[0]
            if not self.coins_picked[coin_idx]  and  not self.coin_holding:
                # pick a coin if 1) it's still there, and 2) I'm not holding another coin
                self.coin_holding = True
                self.coins_picked[coin_idx] = True

        elif next_state_type == 'G':  # goal (coin drop-off area)
            if self.coin_holding:
                # drop the coin
                self.coin_holding = False
                reward = 1
            if np.all(self.coins_picked):
                # finish if all coins have been collected and delivered
                done = True

        elif next_state_type in ['F', 'S']:  # free space
            pass  # default step penalty

        else:
            raise NotImplementedError

        # Return observation and others
        obs = self.get_current_obs()
        return Step(observation=obs,
                    reward=reward,
                    done=done,
                    prev_pos_xy=prev_pos_xy,
                    next_pos_xy=self._get_pos_as_xy()
                   )



    @overrides
    def render(self, mode='human'):
        pass


    @overrides
    def log_diagnostics(self, paths):
        """
        Plot all paths in current batch.
        """
        if self.plot_opts['visitation']:
            self._plot_visitations(paths, self.plot_opts['visitation'])


    def _get_pos_as_xy(self, pos=None):
        """
        Get agent`s position as [X,Y], instead of [row, column].
        :param pos: (r,c) position of agent, or None (current position is used)
        """
        if pos is None:
            pos = self.agent_pos
        r, c = pos
        rows, _ = self.map.shape
        return np.array([c, rows - r - 1])

    def _plot_visitations(self, paths, opts=None):
        """
        Plot visitation graphs, i.e. stacked all paths in batch.
        :param paths: paths statistics (dict)
        :param opts: plotting options:
                {'save': directory to save, True for default directory, or False to disable,
                 'live': <boolean>,
                 'alpha': <0..1> opacity of each plotted path,
                 'noise': <0..1> amount of noise added to distinguish individual paths}
        """
        if opts is None:
            opts = dict()
        if opts.get('live', False):
            plt.figure('Paths')
        else:
            plt.ioff()
        plt.clf()

        # Common plot opts
        m = self.map
        plt.tight_layout()
        plt.xlim(-0.5, self.n_col - 0.5)
        plt.ylim(-0.5, self.n_row - 0.5)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.gca().set_aspect('equal')

        # # Grid
        # x_grid = np.arange(self.n_col + 1) - 0.5
        # y_grid = np.arange(self.n_row + 1) - 0.5
        # plt.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-',
        #          c='k', lw=1, alpha=0.8)
        # plt.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-',
        #          c='k', lw=1, alpha=0.8)

        # Coins, holes, starts, goals and walls
        coins  = self._get_pos_as_xy(np.argwhere(m == 'C').T)
        holes  = self._get_pos_as_xy(np.argwhere(m == 'H').T)
        starts = self._get_pos_as_xy(np.argwhere(m == 'S').T)
        goals  = self._get_pos_as_xy(np.argwhere(m == 'G').T)
        walls  = self._get_pos_as_xy(np.argwhere(m == 'W').T)
        plt.scatter(*coins, c='gold', marker='o', s=150, zorder=10, edgecolors='black')
        plt.scatter(*holes, c='red',  marker='X', s=100, zorder=10)
        plt.gca().add_collection(
            PatchCollection([Rectangle(xy - 0.5, 1, 1) for xy in starts.T],
                            color='navajowhite'))
        plt.gca().add_collection(
            PatchCollection([Rectangle(xy - 0.5, 1, 1) for xy in goals.T],
                            color='lightgreen'))
        plt.gca().add_collection(
            PatchCollection([Rectangle(xy - 0.5, 1, 1) for xy in walls.T],
                            color='navy'))

        # Plot paths
        alpha = opts.get('alpha', 0.1)
        noise = opts.get('noise', 0.1)
        for path in paths:
            data = path['env_infos']
            # Concat subpaths from HRL rollout
            if 'prev_pos_xy' not in data:
                data = SubpolicyPathInfo.concat_subpath_infos(
                    path['env_infos']['subpath_infos']
                )['env_infos']
            # Starting position
            start_pos = data['prev_pos_xy'][:1].T
            # All others
            all_pos = data['next_pos_xy'].T
            all_pos = np.c_[start_pos, all_pos]
            all_pos = all_pos + np.random.normal(size=all_pos.shape, scale=noise)
            # Colorful line collection
            points = all_pos.T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=plt.get_cmap('jet'), alpha=alpha)
            lc.set_array(np.arange(all_pos.shape[-1]))
            plt.gca().add_collection(lc)


        # Save paths figure
        folder = opts.get('save', False)
        if folder:
            if isinstance(folder, str):
                folder = os.path.expanduser(folder)
                if not os.path.isdir(folder):
                    os.makedirs(folder)
            else:
                folder = logger.get_snapshot_dir()
            plt.savefig(os.path.join(folder, 'visitation{:0>3d}.png'.format(
                self.visitation_plot_num)))
            self.visitation_plot_num += 1

        # Live plotting
        if opts.get('live', False):
            plt.gcf().canvas.draw()
            plt.waitforbuttonpress(timeout=0.001)
