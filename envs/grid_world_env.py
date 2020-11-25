import os
import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from gym.spaces import Box, Discrete
from garage.core import Serializable
from garage.envs import Step
from garage.misc.overrides import overrides
from garage.misc import logger

from sandbox.asa.envs import AsaEnv



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
            "SSS.....",
            "SSS.....",
            "SSS#....",
            ".....#..",
            "...#....",
            ".#H...#.",
            ".H..#.#.",
            ".G.#.GGG"
          ]
    STEP_PENALTY = 0.05


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

        # Set state
        self.agent_pos = None
        self.n_row, self.n_col = self.map.shape

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
        Position of agent ++ position of goal
        """
        return Box(low=np.array([0, 0]),
                   high=np.array([self.n_col, self.n_row]) - 1,
                   dtype=np.float32)


    @overrides
    def get_current_obs(self):
        """
        Position of agent
        """
        return np.array(self.agent_pos)


    def reset(self):
        """
        Initialize the agent positioned randomly in one of starting points.
        """
        starts = np.argwhere(self.map == 'S')
        n_starts = starts.shape[0]
        self.agent_pos = starts[np.random.randint(n_starts)]
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
            plt.clf()
        else:
            plt.ioff()

        # Common plot opts
        m = self.map
        plt.tight_layout()
        plt.xlim(-0.5, self.n_col - 0.5)
        plt.ylim(-0.5, self.n_row - 0.5)
        plt.xticks([], [])
        plt.yticks([], [])
        # Grid
        x_grid = np.arange(self.n_row + 1) - 0.5
        y_grid = np.arange(self.n_col + 1) - 0.5
        plt.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-',
                 c='k', lw=1, alpha=0.8)
        plt.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-',
                 c='k', lw=1, alpha=0.8)
        # Start, goal, walls and holes
        starts = self._get_pos_as_xy(np.argwhere(m == 'S').T)
        goals  = self._get_pos_as_xy(np.argwhere(m == 'G').T)
        holes  = self._get_pos_as_xy(np.argwhere(m == 'H').T)
        walls  = self._get_pos_as_xy(np.argwhere(m == 'W').T)
        plt.scatter(*goals, c='royalblue', marker='*', s=100)
        plt.scatter(*holes, c='r', marker='X', s=100)
        plt.gca().add_collection(
            PatchCollection([Rectangle(xy - 0.5, 1, 1) for xy in starts.T],
                            color='navajowhite'))
        plt.gca().add_collection(
            PatchCollection([Rectangle(xy - 0.5, 1, 1) for xy in walls.T],
                            color='navy'))

        # Plot paths
        alpha = opts.get('alpha', 0.1)
        noise = opts.get('noise', 0.1)
        for path in paths:
            # Starting position
            start_pos = path['env_infos']['prev_pos_xy'][:1].T
            # All others
            all_pos = path['env_infos']['next_pos_xy'].T
            all_pos = np.c_[start_pos, all_pos]
            all_pos = all_pos + np.random.normal(size=all_pos.shape, scale=noise)
            plt.plot(all_pos[0], all_pos[1], ls='-', c='darkgreen', alpha=alpha)

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
