import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})



class PlotMakerThesis:
    def __init__(self):
        # Input / output paths
        datafiles = {
            'mb asa':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Top_skill.xlsx',
            'mb ideal':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_sLLLs_disc099_skill.xlsx',
            'mb bad':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Random_skill.xlsx',
            'mb hippo':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS02_Minibot\\All_data--local.xlsx',
            'mb asa partial match': 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST23_Resumed_Partial_match_Top_skill\\All_data--Resumed_with_Top_Euclidean.xlsx',
            'gw asa':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_Top_skill.xlsx',
            'gw ideal':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWTarget_skill.xlsx',
            'gw bad':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWRandom_skill.xlsx',
            'gw hippo':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS03_Gridworld\\All_data--RUNS03_Gridworld.xlsx',
        }
        self.data = DataManager(datafiles, self)
        self.out_folder = r'C:\Users\Juraj\Documents\Škola\FMFI\PhD\Dizertačná práca\img\plots'
        self.out_folder = os.path.join(self.out_folder, datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))

        # General settings
        self.save_plots = False
        if self.save_plots:
            os.mkdir(self.out_folder)

        # Per-environment settings
        self.env_params = {
            'mb': {  # Minibot
                'high_low_percentile': 5,
                'unwanted_seeds': [],
                'true_asa_runs' : [  # (seed, resumed_from_data) tuples
                    (1, 11), (2, 11), (3, 11), (4, 7), (5, 15), (6, 11), (7, 15), (8, 11)
                ],
                'other_asa_runs' : [  # list of resumed_from_data
                    3, 7, 11, 15, 23, 31, 39
                ],
                'plot_x_lim': (0, 79),
                'plot_y_lim': (-2.25, 0),
                'plot_x_tics': (list(range(0, 80, 10)) + [79], list(map(str, range(0, 81, 10)))),
                'plot_y_tics': (np.arange(-2.25, 0.1, 0.25), ['', '$-2.0$', '', '$-1.5$', '', '$-1.0$', '', '$-0.5$', '', '0.0'])
            },
            'gw': {  # Gridworld
                'high_low_percentile': 25,
                'unwanted_seeds': [8, 16],
                'true_asa_runs' : [  # (seed, resumed_from_data) tuples
                    (12, 109), (13, 49), (14, 59), (15, 49), (3, 49), (4, 69), (5, 49), (7, 79)
                ],
                'other_asa_runs' : [  # list of resumed_from_data
                    19, 49, 79, 109, 139
                ],
                'plot_x_lim': (0, 299),
                'plot_y_lim': (0, 6),
                'plot_x_tics': (list(range(0, 300, 50)) + [299], list(map(str, range(0, 301, 50)))),
                'plot_y_tics': (list(range(7)),)
            },
        }
        # TODO sizing of ideal_asa_bad plots
        # TODO sizing of 2nd and 3rd integrators plots


    def do_plots(self):
        self.plot_asa_with_hippo('gw')
        self.plot_asa_with_hippo('mb')
        self.plot_new_skill_usage('gw')
        self.plot_new_skill_usage('mb')
        self.plot_ideal_asa_bad_selected('gw')
        self.plot_ideal_asa_bad_selected('mb')
        self.plot_ideal_asa_bad_all('gw')
        self.plot_ideal_asa_bad_all('mb')
        self.plot_integrators(3, labels=True)
        self.plot_integrators(11)
        self.plot_integrators(39)


    ################ Plotters ################
    def plot_asa_with_hippo(self, env):
        """
        Plot true ASA run compared with HiPPO
        """
        print(f'Plotting ASA with HiPPO - {env}')
        plt.figure()
        params = self.env_params[env]
        basic_run_split = max(11, min([itr for _, itr in params['true_asa_runs']]))

        # Basic run
        self.draw_reward_range(
                self.data(env, 'asa')
                    .filter_basic_runs()
                    .filter_itr_to(basic_run_split),
                env, color='darkblue'
        )
        self.draw_reward_range(
                self.data(env, 'asa')
                    .filter_basic_runs()
                    .filter_itr_from(basic_run_split),
                env, color='black', label='Base run'
        )

        # True ASA runs
        self.draw_reward_range(
                self.data(env, 'asa')
                    .filter_seed_and_resumed_from(params['true_asa_runs'])
                    .filter_itr_from(12)
                    .append_prev_itr(),
                env, color='royalblue', label='With ASA skill'
        )

        # HiPPO runs
        hippo_colors = ['darkorange', 'orangered']
        hippo_exp_names = {
            'mb': ['latdim3_period3_5', 'latdim10_period3_5'],
            'gw': ['latdim14_period2_23', 'latdim50_period2_23']
        }
        hippo_labels = {
            'mb': ['HiPPO - latent 3', 'HiPPO - latent 10'],
            'gw': ['HiPPO - latent 18', 'HiPPO - latent 50']
        }
        hippo_unwanted_seeds = {
            'mb': [],
            'gw': [10, 40, 50, 100]
        }
        for exp_name, label, color in zip(hippo_exp_names[env], hippo_labels[env], hippo_colors):
            self.draw_reward_range(
                    self.data(env, 'hippo', filter_unwanted_seeds=False)
                        .filter_unwanted_seeds(hippo_unwanted_seeds[env])
                        .filter_exp_name(exp_name),
                    env, color=color, label=label
            )

        # Finalize
        self.tidy_plot(env, w=24, h=12)
        plt.legend()
        self.show_save_plot(f'asa-with-hippo-{env}')


    def plot_asa_individual_runs(self, env):
        """
        Plot a) individual runs (seeds) of true ASA, b) new skill usage
        """
        print(f'Plotting ASA - individual runs - {env}')
        params = self.env_params[env]
        plt.figure()
        seeds = sorted(list({int(row['ExpSeed'][1:]) for row in self.data(env, 'asa').get_all()}))
        colors = [f'C{i}' for i in range(10)]
        # DEBUG: custom seed order and colors - only working for GW
        seeds = [5, 3, 14, 15, 13, 4, 12, 7]
        cmap = matplotlib.cm.get_cmap('gist_rainbow')
        colors = [cmap(x) for x in np.mod(np.linspace(0, 0.9, len(seeds))+0.75, 1)]

        # Plot for each seed
        for i, seed in enumerate(seeds):
            seed_data = self.data(env, 'asa')\
                    .filter_seed(seed)\
                    .save_as_default()
            plt.plot(
                *seed_data.default()
                    .filter_basic_runs()
                    .get_reward_mean(smooth=0.75),
                color=colors[i],
                # alpha=0.5,
                linewidth=1,
                linestyle=':'
            )
            plt.plot(
                *seed_data.default()
                    .filter_seed_and_resumed_from(params['true_asa_runs'])
                    .append_prev_itr()
                    .get_reward_mean(smooth=0.75),
                color=colors[i],
                linewidth=1.25,
                label=f's{seed}'
            )

        # Finalize
        self.tidy_plot(env, w=12, h=12)
        plt.legend()
        self.show_save_plot(f'asa-individual-runs-{env}')


    def plot_new_skill_usage(self, env):
        """
        Plot new skills' usage
        """
        print(f'Plotting new skills\' usage - {env}')
        params = self.env_params[env]
        plt.figure()
        seeds = sorted(list({int(row['ExpSeed'][1:]) for row in self.data(env, 'asa').get_all()}))
        colors = ['#a65628', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#999999', '#f781bf', '#e41a1c'][::-1]

        # Plot for each seed
        for i, seed in enumerate(seeds):
            seed_data = self.data(env, 'asa')\
                    .filter_seed_and_resumed_from(params['true_asa_runs'])\
                    .filter_seed(seed)\
                    .save_as_default()
            _, batch_size = seed_data.get_aggr('BatchSamples')
            batch_size = 10000 - batch_size
            skill_attr = '_19' if env == 'mb' else '_34'
            itrs, skill_usage = seed_data.get_aggr(skill_attr, smooth=0.7 * (env == 'gw'))
            plt.plot(
                itrs, skill_usage * batch_size,
                color=colors[i],
                label=f'seed {i+1}'
            )

        # Finalize
        y_max = 5000 if env == 'mb' else 2250
        legend_loc = 'lower right' if env == 'mb' else 'upper right'
        if env == 'gw':
            plt.yticks(range(0, 2251, 250), ['0', '', '500', '', '1000', '', '1500', '', '2000', ''])
        y_label = 'Invocations of new skill' if env == 'gw' else None
        self.tidy_plot(env, w=12, h=10, y_lim=(0, y_max), y_label=y_label)
        plt.legend(ncol=2, loc=legend_loc)
        self.show_save_plot(f'asa-individual-runs-{env}')


    def plot_ideal_asa_bad_selected(self, env):
        """
        Plot performance of ideal, ASA, and bad skill from true ASA runs
        """
        print(f'Plotting ideal, ASA, and bad skill - selected runs - {env}')
        params = self.env_params[env]
        plt.figure()


        # Basic run
        self.draw_reward_range(
                self.data(env, 'asa')
                    .filter_basic_runs(),
                env, color='black', label='Base run'
        )

        # With skills
        skills = [
            ('ideal', 'With ideal skill', '#2BAB2B'),
            ('asa', 'With ASA skill', 'royalblue'),
            ('bad', 'With bad skill', '#FF3333')
        ]
        for dataset, label, color in skills:
            self.draw_reward_range(
                    self.data(env, dataset)
                        .filter_seed_and_resumed_from(params['true_asa_runs'])
                        .filter_itr_from(12)
                        .append_prev_itr(),
                    env, color=color, label=label
            )


        # Finalize
        y_label = y_label='Average discounted reward' if env == 'gw' else None
        self.tidy_plot(env, w=12, h=10, y_label=y_label)
        plt.legend(loc='lower right')
        self.show_save_plot(f'ideal-asa-bad-selected-{env}')


    def plot_ideal_asa_bad_all(self, env):
        """
        Plot performance of ideal, ASA, and bad skill from true ASA runs
        """
        print(f'Plotting ideal, ASA, and bad skill - all runs in subplots - {env}')
        rcParams.update({'figure.autolayout': False})
        params = self.env_params[env]
        fig, _ = plt.subplots(3, 1, sharex='all', sharey='all')
        fig.subplots_adjust(hspace=0, left=0.12, right=0.97, top=0.99, bottom=0.05)

        # Plot for all three skills
        # color tool: https://www.cssfontstack.com/oldsites/hexcolortool/
        skills = [
            (1, 'ideal', 'Manually added ideal skill', '#2BAB2B', ['#61E161', '#4FCF4F', '#3DBD3D', '#2BAB2B', '#199919', '#078707', '#007500']),
            (2, 'asa', 'Manually triggered ASA', '#4464C9',     ['#91B1FF', '#7797FC', '#5E7EE3', '#4464C9', '#2B4BB0', '#113196', '#00187D']),
            (3, 'bad', 'Manually added bad skill', '#FF3333',     ['#FF8F8F', '#FF7070', '#FF5252', '#FF3333', '#ED2121', '#DB0F0F', '#C90000'])
        ]
        for subplot, dataset, label, main_color, colors in skills:
            plt.subplot(3, 1, subplot)
            # Basic run
            plt.plot(
                *self.data(env, 'asa')
                    .filter_basic_runs()
                    .get_reward_mean(),
                color='black'
            )
            # With skills
            if env == 'gw':
                colors = colors[:1] + colors[2:-2] + colors[-1:]
            for i, itr in enumerate(params['other_asa_runs']):
                plt.plot(
                        *self.data(env, dataset)
                            .filter_resumed_from(itr)
                            .append_prev_itr()
                            .get_reward_mean(),
                        color=colors[i]
                )
            plt.plot(-1, -1, color=main_color, label=label)

            # Format subplot
            plt.legend(loc='lower right')
            plt.grid()
            plt.xlim(*self.env_params[env]['plot_x_lim'])
            if subplot < 3:
                plt.gca().xaxis.set_ticklabels([])
            plt.ylim(*self.env_params[env]['plot_y_lim'])
            y_tics = list(self.env_params[env]['plot_y_tics'])
            y_tics[0] = y_tics[0][:-1]
            plt.yticks(*y_tics)
            if env == 'gw'  and  subplot == 2:
                plt.ylabel('Average discounted reward')

        # Finalize
        w, h = 12, 25
        plt.gcf().set_size_inches(w=w/2.54, h=h/2.54)   # convert to inches
        plt.xticks(*self.env_params[env]['plot_x_tics'])
        plt.xlabel('Iteration')
        self.show_save_plot(f'ideal-asa-bad-selected-{env}')
        rcParams.update({'figure.autolayout': True})


    def plot_integrators(self, res_from, labels=False):
        env = 'mb'
        print(f'Plotting integrators - {env} - resumed from {res_from}')
        plt.figure()

        # Basic run
        plt.plot(
                *self.data(env, 'asa')
                    .filter_basic_runs()
                    .get_reward_mean(),
                color='black', label='Base run'
        )

        # ASA with different integrators
        integrators = [
            ('rnd',         'Random',           'red'),
            ('rndBias',     'Bias–boosted',     'darkorange'),  # labels have en-dashes
            ('startObsAvg', 'Start–states',     'blue'),
            ('sbptFrst',    'First–skill',      'green'),
            ('sbptAvg',     'All–skills',       'tab:cyan'),
            ('sbptSmthAvg', 'Smoothed–skills',  'purple')
        ]
        for integrator, label, color in integrators:
            kwargs = dict()
            if not integrator.startswith('rnd'):
                kwargs = {'linewidth': 0.8, 'alpha': 0.8}
            plt.plot(
                *self.data(env, 'asa')
                    .filter_skill_name('Top')
                    .filter_integrator(integrator)
                    .filter_resumed_from(res_from)
                    .append_prev_itr()
                    .get_reward_mean(),
                color=color,
                **kwargs
            )

        # Finalize
        y_label = 'Average discounted reward' if labels else None
        self.tidy_plot(env, w=12, h=10, y_label=y_label, y_hide_numbers=not labels)
        self.show_save_plot(f'integrators-{env}-from-itr{res_from+1}')

        # Legend
        if labels:
            print(f'Plotting legend for integrators')
            plt.figure(figsize=(4.52, 1.18))
            plt.axis('off')
            plt.plot(0, 0, color='black', label='Base run')
            plt.plot(0, 0, alpha=0, label=' ')
            plt.plot(0, 0, alpha=0, label=r'$\bf{Uninformed\ schemes:}$')
            for _, label, color in integrators[:2]:
                plt.plot(0, 0, color=color, label=label)
            plt.plot(0, 0, alpha=0, label=r'$\bf{Informed\ schemes:}$')
            for _, label, color in integrators[2:]:
                plt.plot(0, 0, color=color, label=label)
            plt.legend(ncol=2, loc='center', bbox_to_anchor=(0.1, 0.5))
            self.show_save_plot(f'integrators-legend')



    ################ Helpers ################
    def draw_reward_range(self, data_builder, env, color=None, alpha=0.2, **kwargs):
        """
        Plot range of reward - high and low percentile
        """
        x, mean = data_builder.get_reward_mean()
        _, low  = data_builder.get_reward_percentile(self.env_params[env]['high_low_percentile'])
        _, high = data_builder.get_reward_percentile(100 - self.env_params[env]['high_low_percentile'])
        plt.plot(x, mean, c=color, **kwargs)
        plt.fill_between(x, low, high, color=color, alpha=alpha)

    def tidy_plot(self, env, w=24., h=10.,
                  x_label='Iteration', y_label='Average discounted reward',
                  x_lim=None, y_lim=None, y_hide_numbers=False,
                  tight_layout=True
                 ):
        """
        Produce a nice and tidy plot
        """
        w, h = w/2.54, h/2.54   # convert to inches
        plt.gcf().set_size_inches(w=w, h=h)
        if tight_layout:
            plt.tight_layout()
        plt.grid()
        if x_lim is None:
            x_lim = self.env_params[env]['plot_x_lim']
            plt.xticks(*self.env_params[env]['plot_x_tics'])
        plt.xlim(*x_lim)
        if y_lim is None:
            y_lim = self.env_params[env]['plot_y_lim']
            plt.yticks(*self.env_params[env]['plot_y_tics'])
        plt.ylim(*y_lim)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if y_hide_numbers:
            plt.gca().set_yticklabels([])

    def show_save_plot(self, name=''):
        """
        Show plot, and save it if save_plots option is set to True
        """
        if self.save_plots:
            plt.savefig(os.path.join(self.out_folder, f'{name}.png'), dpi=300)
        plt.show()



class DataManager:
    def __init__(self, datafiles, plotter):
        """
        Initialize data manager with given data files
        :param datafiles: dict name_of_dataset -> path_to_file
        """
        self.datafiles = datafiles
        self.datasets = dict()
        self.plotter = plotter

    def ensure_dataset_loaded(self, name):
        """
        Load dataset into memory, if not already done
        """
        if name not in self.datasets:
            print(f'Loading dataset "{name}"')
            pd_data = pd.read_excel(self.datafiles[name])
            data = pd.DataFrame.to_dict(pd_data, 'records')
            self.datasets[name] = data

    def builder(self, *dataset_name_parts, filter_unwanted_seeds=True):
        dataset_name = ' '.join(dataset_name_parts)
        env = dataset_name.split()[0]

        self.ensure_dataset_loaded(dataset_name)
        builder = DatasetBuilder(self.datasets[dataset_name])
        if filter_unwanted_seeds:
            builder = builder\
                    .filter_unwanted_seeds(self.plotter.env_params[env]['unwanted_seeds'])\
                    .save_as_default()
        return builder

    def __call__(self, *dataset_name_parts, filter_unwanted_seeds=True):
        return self.builder(*dataset_name_parts, filter_unwanted_seeds=filter_unwanted_seeds)



class DatasetBuilder:
    def __init__(self, data):
        self.default_data = data
        self.data = list(data)


    def get_all(self):
        """
        Get a working copy of whole dataset (list of dicts)
        """
        self.data = list(self.data)
        return self.data

    def get_aggr(self, attribute, aggregator=None, smooth=0., **kwargs):
        """
        Get aggregation from desired attribute. Default aggregation is mean.
        Optionally apply exponential smoothing average on result.
        Returns tuple (iterations, values)
        """
        if aggregator is None:
            aggregator = np.mean
        data = self.get_all()
        itrs = {row['Iteration'] for row in data}
        itrs = sorted(list(itrs))
        vals = []
        running_avg = 0
        for itr in itrs:
            itr_data = DatasetBuilder(data).filter_itr(itr).get_all()
            val = aggregator([row[attribute] for row in itr_data], **kwargs)
            if len(vals) == 0:
                running_avg = val
            else:
                running_avg = smooth * running_avg + (1 - smooth) * val
            vals.append(running_avg)
        return np.array(itrs), np.array(vals)

    def get_reward_mean(self, **kwargs):
        return self.get_aggr('AverageDiscountedReturn', **kwargs)

    def get_reward_percentile(self, q, **kwargs):
        return self.get_aggr('AverageDiscountedReturn', np.percentile, q=q, **kwargs)

    def count(self):
        return len(self.get_all())

    def save_as_default(self):
        """
        Save current state as default, apply new modifications always on top of current state
        """
        self.default_data = self.get_all()
        return self

    def default(self):
        self.data = self.default_data
        return self


    def append_prev_itr(self):
        """
        Append data of previous iteration from Basic runs to dataset
        """
        self.data = self.get_all()  # need to materialize, since filter can be consumed only once
        min_itr = min([row['Iteration'] for row in self.data])
        basic_runs = DatasetBuilder(self.default_data)\
            .filter_basic_runs()\
            .filter_itr(min_itr - 1)\
            .get_all()
        self.data.extend(basic_runs)
        return self


    def filter_basic_runs(self):
        self.data = filter(
            lambda row: row['ExpName'].startswith('Basic_run'),
            self.data)
        return self

    def filter_seed(self, seed):
        self.data = filter(
            lambda row: row['ExpSeed'] == f's{seed}',
            self.data)
        return self

    def filter_exp_name(self, exp_name):
        self.data = filter(
            lambda row: row['ExpName'] == exp_name,
            self.data)
        return self

    def filter_skill_name(self, skill_name, this_or_others=True):
        # Return this skill if this_or_others==True, or all other skills if this_or_others=False
        self.data = filter(
            lambda row: (skill_name in row['ExpSkill']) == this_or_others,
            self.data)
        return self

    def filter_resumed_from(self, itr):
        self.data = filter(
            lambda row: row['ExpResumedFrom'] == itr,
            self.data)
        return self

    def filter_integrator(self, integrator):
        self.data = filter(
            lambda row: row['ExpIntegrator'] == integrator,
            self.data)
        return self

    def filter_unwanted_seeds(self, unwanted_seeds):
        def filter_func(row):
            s = row['ExpSeed']
            if type(s) is str:
                return int(s[1:]) not in unwanted_seeds
            else:
                return s not in unwanted_seeds
        self.data = filter(filter_func, self.data)
        return self

    def filter_seed_and_resumed_from(self, seed_itr_list):
        self.data = filter(
            lambda row: (int(row['ExpSeed'][1:]), row['ExpResumedFrom'])  in  seed_itr_list,
            self.data)
        return self

    def filter_itr(self, itr):
        self.data = filter(
            lambda row: row['Iteration'] == itr,
            self.data)
        return self

    def filter_itr_from(self, itr):
        self.data = filter(
            lambda row: row['Iteration'] >= itr,
            self.data)
        return self

    def filter_itr_to(self, itr):
        self.data = filter(
            lambda row: row['Iteration'] <= itr,
            self.data)
        return self



if __name__ == "__main__":
    plot_maker = PlotMakerThesis()
    plot_maker.do_plots()
    print('Done!')
