import numpy as np
import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt

from garage.tf.algos.batch_polopt import BatchPolopt
from garage.core.serializable import Serializable
from garage.misc.overrides import overrides
import garage.misc.logger as logger

from sandbox.asa.tools.path_trie import PathTrie


class ASAWrapper(BatchPolopt):

    def __init__(self,
                 env,
                 policy,
                 baseline,
                 top_algo_cls,
                 asa_plot,
                 **kwargs):
        """
        Wrapper for a top-level RL algorithm that performs Adaptive Skill Acquisition in HRL.
        :param env: environment
        :param policy: policy
        :param baseline: baseline
        :param top_algo_cls: class of RL algorithm for training top-level agent. Must inherit BatchPolopt (only
                             init_opt(), optimize_policy(), and get_itr_snapshot() will be used).
        :param asa_plot: which plots to generate:
                {'visitation': <opts>, 'aggregation': <opts>}
                where opts = {'save': <directory or False>, 'live': <boolean> [, 'alpha': <0..1>][, 'noise': <0..1>]}
        """
        self._top_algo = top_algo_cls(env=env,
                                      policy=policy,
                                      baseline=baseline,
                                      **kwargs)
        super().__init__(env=env,
                         policy=policy,
                         baseline=baseline,
                         **kwargs)
        self.sampler = self._top_algo.sampler
            
        # Plotting
        self.visitation_plot_num = 0
        self.aggregation_plot_num = 0
        if (asa_plot is None) or (asa_plot == False):
            self.plot_opts = {}
        else:
            self.plot_opts = asa_plot
        for plot_type in ['visitation', 'aggregation']:
            if not plot_type in self.plot_opts  or  not isinstance(self.plot_opts[plot_type], dict):
                self.plot_opts[plot_type] = {}
        if any([plot_type_opts.get('live', False) for plot_type_opts in self.plot_opts.values()]):
            plt.ion()

    @overrides
    def init_opt(self):
        res = self._top_algo.init_opt()
        return res

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        res = self._top_algo.get_itr_snapshot(itr, samples_data)
        # TODO? res['some hrl stuff'] = None
        return res

    @overrides
    def optimize_policy(self, itr, samples_data):
        self._top_algo.optimize_policy(itr, samples_data)

    @overrides
    def log_diagnostics(self, paths):
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)
        self.asa_log(paths)

    def asa_log(self, paths):
        """
        Log ASA stuff: count frequent subpaths, log top hits, plot all paths, plot aggregated paths.
        """
        with logger.prefix('ASA | '):
            # Trie parameters
            min_length = 3
            max_length = 10
            action_map = {0: 'L', 1: 's'}
            min_f_score = 2  # 20
            max_results = 10
            aggregations = ['mean']  # sublist of ['mean', 'most_freq', 'nearest_mean', 'medoid'] or 'all'

            # Count paths
            path_trie = PathTrie(self.env.action_space.n)
            for path in paths:
                actions = path['actions'].argmax(axis=1).tolist()
                observations = path['observations']
                path_trie.add_all_subpaths(actions,
                                           observations,
                                           min_length=min_length,
                                           max_length=max_length)
            logger.log('Searched {} rollouts'.format(len(paths)))

            frequent_paths = path_trie.items(
                action_map=action_map,
                min_count=len(paths) * 2,   # TODO? what about this?
                min_f_score=min_f_score,
                max_results=max_results,
                aggregations=aggregations
            )
            logger.log('Found {} frequent paths: [actions, count f-score]'.format(len(frequent_paths)))
            for f_path in frequent_paths:
                logger.log('    {:{pad}}\t{}\t{:.3f}'.format(
                    f_path['actions_text'],
                    f_path['count'],
                    f_path['f_score'],
                    pad=max_length))
        
        # Plots
        if self.plot_opts['visitation']:
            self.plot_visitations(paths, self.plot_opts['visitation'])
        if self.plot_opts['aggregation']:
            self.plot_opts['aggregation']['aggregations'] = aggregations
            self.plot_aggregations(frequent_paths, self.plot_opts['aggregation'])


    def plot_visitations(self, paths, opts={}):
        '''
        Plot visitation graphs, i.e. stacked paths for each map.
        :param paths: paths statistics (dict)
        :param opts: plotting options:
                {'save': <directory or False>,
                 'live': <boolean>,
                 'alpha': <0..1, opacity of each plotted path>,
                 'noise': <0..1, amount of noise added to distinguish individual paths>}  
        '''
        env = self.env.unwrapped
        sbp_count = len(env.maps)
        sbp_nrows = int(np.round(np.sqrt(sbp_count)))
        sbp_ncols = int((sbp_count-1) // sbp_nrows + 1)
        plt.figure('Paths')
        plt.clf()
        fig, ax = plt.subplots(sbp_nrows, sbp_ncols, num='Paths', squeeze=False)
        ax = ax.flatten()
        
        # Plot cells grid
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle
        plt.tight_layout()
        for map_idx, map_ax in enumerate(ax):
            if map_idx >= len(env.maps): map_ax.set_axis_off(); continue
            m = env.maps[map_idx]
            rows, cols = m.shape
            map_ax.set_xlim(-0.5, cols - 0.5)
            map_ax.set_ylim(-0.5, rows - 0.5)
            # Grid
            x_grid = np.arange(rows + 1) - 0.5
            y_grid = np.arange(cols + 1) - 0.5
            map_ax.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='k', lw=1, alpha=0.8)
            map_ax.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='k', lw=1, alpha=0.8)
            # Start, goal, walls and holes
            start = env.get_pos_as_xy(  np.argwhere(m == 'S').T  , rows)
            goal  = env.get_pos_as_xy(  np.argwhere(m == 'G').T  , rows)
            holes = env.get_pos_as_xy(  np.argwhere(m == 'H').T  , rows)
            walls = env.get_pos_as_xy(  np.argwhere(m == 'W').T  , rows)            
            map_ax.scatter(*start, c='r', marker='o', s=50 )
            map_ax.scatter(*goal,  c='r', marker='x', s=50 )
            map_ax.scatter(*holes, c='k', marker='v', s=100)
            map_ax.add_collection(PatchCollection([Rectangle(xy-0.5, 1, 1) for xy in walls.T], color='navy'))
        
        # Plot paths
        alpha = opts.get('alpha', 0.1)
        noise = opts.get('noise', 0.1)
        for path in paths:
            # Starting position
            map_idx = path['env_infos']['map'][0]
            m = env.maps[map_idx]
            (start_r,), (start_c,) = np.nonzero(m == 'S')
            start_pos_rc = np.array([start_r, start_c])
            start_pos_xy = env.get_pos_as_xy(pos=start_pos_rc, rows=m.shape[0])
            # All others
            pos = path['env_infos']['pos_xy'].T
            pos = np.c_[start_pos_xy, pos]
            pos = pos + np.random.normal(size=pos.shape, scale=noise)
            c = env.map_colors[map_idx % len(env.map_colors)]
            ax[map_idx].plot(pos[0], pos[1], ls='-', c=c, alpha=alpha)
        
        # Save paths figure
        dir = opts.get('save', False)
        if dir:
            if isinstance(dir, str):
                dir = os.path.expanduser(dir)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
            else:
                dir = logger.get_snapshot_dir()
            plt.savefig(os.path.join(dir, 'visitation{:0>3d}.png'.format(self.visitation_plot_num)))
            self.visitation_plot_num += 1
        
        # Live plotting
        if opts.get('live', False):
            plt.gcf().canvas.draw()
            plt.waitforbuttonpress(timeout=0.001)
        
    
    def plot_aggregations(self, frequent_paths, opts={}):
        '''
        Plot aggregation graphs, i.e. aggregated start and end for each frequent subpath from trie.
        :param frequent_paths: subpaths with statistics (list obtained by PathTrie.items())
        :param opts: plotting options: {'save': <directory or False>, 'live': <boolean>, 'aggregations': <'all' or list>}  
        '''
        env = self.env.unwrapped
        agg_names = opts['aggregations']
        if agg_names == 'all':
            agg_names = PathTrie.all_aggregations
        if not isinstance(agg_names, list):
            agg_names = [agg_names]
        n_paths = len(frequent_paths)
        n_aggs = len(agg_names)
        
        # Prepare figure and GridSpecs
        fig = plt.figure('Aggregations', figsize=(12,5))
        plt.clf()
        side = 0.025
        gap = 0.015
        gss = [plt.GridSpec(n_aggs, 2*n_paths, left = side+gap*n, right = (1-side)-gap*(n_paths-n), wspace=0.05) for n in range(n_paths)]
        
        # Plot aggregations
        from matplotlib.pyplot import cm
        for ai, agg_name in zip(range(n_aggs), agg_names):
                for pi in range(n_paths):
                        # Construct start and end
                        aggregated = frequent_paths[pi]['agg_observations'][agg_name]
                        split = len(aggregated) // 2
                        start_obs = aggregated[:split].reshape(env.obs_wide, env.obs_wide)
                        end_obs   = aggregated[split:].reshape(env.obs_wide, env.obs_wide)
                        # Prepare axes and plot start and end
                        ax_s = fig.add_subplot(gss[pi][ai,pi*2])
                        ax_e = fig.add_subplot(gss[pi][ai,pi*2+1])
                        ax_s.imshow(start_obs, interpolation='nearest', cmap=cm.Blues, origin='upper')
                        ax_e.imshow(end_obs  , interpolation='nearest', cmap=cm.Blues, origin='upper')
                        # Grid, labels and ticks
                        x_grid = np.arange(env.obs_wide + 1) - 0.5
                        y_grid = np.arange(env.obs_wide + 1) - 0.5
                        for ax in [ax_s, ax_e]:
                            ax.plot(x_grid, np.stack([y_grid] * x_grid.size), ls='-', c='k', lw=1, alpha=0.4)
                            ax.plot(np.stack([x_grid] * y_grid.size), y_grid, ls='-', c='k', lw=1, alpha=0.4)
                            ax.set_xticks([])
                            ax.set_yticks([])
                        if pi == 0: ax_s.set_ylabel(agg_name, size='medium')
                        if ai == 0: ax_s.set_title('{:s}\nf={:.3f}\n#={:d}'.format(
                                    frequent_paths[pi]['actions_text'],
                                    frequent_paths[pi]['f_score'],
                                    frequent_paths[pi]['count'],
                                    ),
                                loc='left')
                    
        # Save paths figure
        dir = opts.get('save', False)
        if dir:
            if isinstance(dir, str):
                dir = os.path.expanduser(dir)
                if not os.path.isdir(dir):
                    os.makedirs(dir)
            else:
                dir = logger.get_snapshot_dir()
            plt.savefig(os.path.join(dir, 'aggregation{:0>3d}.png'.format(self.aggregation_plot_num)))
            self.aggregation_plot_num += 1
        
        # Live plotting
        if opts.get('live', False):
            plt.gcf().canvas.draw()
            plt.waitforbuttonpress(timeout=0.001)
