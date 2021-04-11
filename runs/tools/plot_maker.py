import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


## Input files
data_files = [
    # Minibot
    # 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Top_skill.xlsx',
    # 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_sLLLs_disc099_skill.xlsx',
    # 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Random_skill.xlsx',
    # 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS02_Minibot\\All_data--local.xlsx',
    # Gridworld
    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_Top_skill.xlsx',
    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWTarget_skill.xlsx',
    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWRandom_skill.xlsx',
    # 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS03_Gridworld\\Latdim14_data--RUNS03_Gridworld.xlsx',
]

## Input params
save_plot = False
save_plot_dir = 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Other\\ICANN 2021\\plots_v2'
do_legends = False
skill_label = 'bad'

## Per-environment params
env = 'minibot' if 'TEST07' in data_files[0] else 'gridworld'
if env == 'minibot':
    # Minibot
    high_low_percentile = 5
    plot_x_lim = (0, 79)
    plot_y_lim = (-2.2, 0)
    unwanted_seeds = []
    true_asa_runs = [   # (seed, resumed_from_data) tuples
        (1, 11), (2, 11), (3, 11), (4, 11), (5, 15), (6, 11), (7, 15), (8, 11)
    ]
    other_asa_runs = [  # list of resumed_from_data
        3, 7, 11, 15, 23, 31, 39
    ]
    ideal_bad_skill_names = ['sLLLs', 'Random']
    hippo_exp_names = ['latdim3_period3_5', 'latdim10_period3_5']
    hippo_labels = ['HiPPO - latent 3', 'HiPPO - latent 10']
    hippo_unwanted_seeds = []
else:
    # Gridworld
    high_low_percentile = 25
    plot_x_lim = (0, 299)
    plot_y_lim = (0, 6)
    unwanted_seeds = [8, 16]
    true_asa_runs = [   # (seed, resumed_from_data) tuples
        (12, 109), (13, 49), (14, 59), (15, 49), (3, 49), (4, 69), (5, 49), (7, 79)
    ]
    other_asa_runs = [  # list of resumed_from_data
        19, 49, 79, 109, 139
    ]
    ideal_bad_skill_names = ['Target', 'Random']
    hippo_exp_names = ['latdim14_period2_23', 'latdim50_period2_23']
    hippo_labels = ['HiPPO - latent 18', 'HiPPO - latent 50']
    hippo_unwanted_seeds = [40, 50, 80, 100]





## Filters
lfilter = lambda f, d: list(filter(f, d))
f_basic_run      = lambda d: lfilter(lambda r: r['ExpName'].startswith('Basic_run'), d)
f_exp_name       = lambda d, str: lfilter(lambda r: str == r['ExpName'], d)
f_skill_name     = lambda d, str: lfilter(lambda r: str in r['ExpSkill']  or  r['ExpSkill'] == '-', d)
f_not_skill_name = lambda d, str: lfilter(lambda r: str not in r['ExpSkill'], d)
f_resumed_from   = lambda d, itr: lfilter(lambda r: r['ExpResumedFrom'] == itr, d)
f_seed_resfrom   = lambda d, s, itr: lfilter(lambda r: r['ExpSeed'] == f's{s}'  and  r['ExpResumedFrom'] == itr, d)
f_itr            = lambda d, itr: lfilter(lambda r: r['Iteration'] == itr, d)
f_itr_from       = lambda d, itr: lfilter(lambda r: r['Iteration'] >= itr, d)
f_itr_to         = lambda d, itr: lfilter(lambda r: r['Iteration'] <= itr, d)
f_integrator     = lambda d, integ: lfilter(lambda  r: r['ExpIntegrator'] == integ, d)


## Aggregators
def get_stat(data, attribute, aggr_func, **kwargs):
    """
    Get a statistical measure (given by aggregation function) of given attribute
    """
    itrs = {r['Iteration'] for r in data}
    itrs = sorted(list(itrs))
    vals = []
    for itr in itrs:
        vals.append(aggr_func([r[attribute] for r in f_itr(data, itr)], **kwargs))
    return np.array([itrs, vals])

get_reward_mean = lambda d: get_stat(d, 'AverageDiscountedReturn', np.mean)                                     # Get mean
get_reward_low  = lambda d: get_stat(d, 'AverageDiscountedReturn', np.percentile, q=high_low_percentile)        # Get low percentile
get_reward_high = lambda d: get_stat(d, 'AverageDiscountedReturn', np.percentile, q=100 - high_low_percentile)  # Get high percentile


## Helpers
def append_prev_itr(data, basic_run_data):
    """
    Append data of previous iteration from Basic runs to dataset
    """
    min_itr = min({r['Iteration'] for r in data})
    res = list(data)
    res.extend(f_itr(basic_run_data, min_itr-1))
    return res


def plot_range(data, color=None, label=None, alpha=0.2, **kwargs):
    """
    Plot range of reward - high and low percentile
    """
    X, M  = get_reward_mean(data)
    _, Q1 = get_reward_low(data)
    _, Q3 = get_reward_high(data)
    plt.plot(X, M, c=color, label=label, **kwargs)
    plt.fill_between(X, Q1, Q3, color=color, alpha=alpha)


def tidy_plot(y_label='Avg. discounted reward', w=4, h=3):
    """
    Produce a nice and tidy plot
    """
    plt.gcf().set_size_inches(w=w, h=h)
    plt.tight_layout()
    plt.grid()
    plt.xlim(*plot_x_lim)
    plt.ylim(*plot_y_lim)
    plt.xlabel('Iteration')
    plt.ylabel(y_label if y_label else None)
    if y_label is None:
        plt.gca().set_yticklabels([])
        plt.gcf().set_size_inches(w=w-0.5, h=h-0.06)


def text_to_plot(text, x=0.04, y=0.87, fontsize=16, **kwargs):
    plt.text(x, y, text,
             transform=plt.gca().transAxes,
             fontsize=fontsize,
             **kwargs
    )


def show_save_plot(name=''):
    """
    Show plot, and save it if save_plot option is set to True
    """
    if save_plot:
        dt = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        plt.savefig(os.path.join(save_plot_dir, f'{dt}-{name}-{env}.png'), dpi=300)
    plt.show()



## Data gathering
pd_data = pd.concat(
    [pd.read_excel(f) for f in data_files],
    ignore_index=True, sort=False
)
data = pd.DataFrame.to_dict(pd_data, 'records')
data = lfilter(
    lambda r: (type(r['ExpSeed']) is str and int(r['ExpSeed'][1:]) not in unwanted_seeds) or (type(r['ExpSeed']) is int and r['ExpSeed'] not in hippo_unwanted_seeds),
    data
)

basic_runs_data = f_basic_run(data)

resumed_from_data = {}
for i in range(plot_x_lim[1]):
    try:
        run = f_resumed_from(data, i)
        if env == 'gridworld' and skill_label == 'ASA':
            run = f_exp_name(run, 'From_T20')
        resumed_from_data[i] = append_prev_itr(run, basic_runs_data)
    except ValueError:
        pass

true_asa_runs_data = []
for s, itr in true_asa_runs:
    true_asa_runs_data.extend(f_seed_resfrom(data, s, itr))
true_asa_runs_data = append_prev_itr(true_asa_runs_data, basic_runs_data)

other_asa_runs_data = [resumed_from_data[itr] for itr in other_asa_runs]





## Plotting

def plot_asa_with_hippo():
    # Plot true ASA run compared with HiPPO
    plt.figure()

    basic_run_split = min([i for _, i in true_asa_runs])
    plot_range(f_itr_to(  basic_runs_data, basic_run_split), color='#7F0000')
    plot_range(f_itr_from(basic_runs_data, basic_run_split), color='k', label='base run')

    plot_range(f_skill_name(true_asa_runs_data, 'Top'), color='r', label='with ASA skill')

    hippo_colors = ['mediumblue', 'darkviolet']
    for hippo_exp, l, c in zip(hippo_exp_names, hippo_labels, hippo_colors):
        plot_range(f_exp_name(data, hippo_exp), color=c, label=l)

    tidy_plot()
    text_to_plot('(a)')
    show_save_plot('asa-with-hippo')
    if do_legends:
        plt.figure()
        plt.gcf().set_size_inches(w=10, h=10)
        plt.plot(0, 0, color='k', label='Base run')
        plt.plot(0, 0, color='r', label='With ASA skill')
        for l, c in zip(hippo_labels, hippo_colors):
            plt.plot(0, 0, color=c, label=l)
        plt.plot(0, 0, color='#009F00', label='Manually triggered ASA')
        plt.legend(bbox_to_anchor=(1, 1.1), ncol=5)
        show_save_plot(f'asa-with-hippo-legend')


def plot_asa_with_ideal_bad_skills():
    # Plot true ASA run compared with ideal/bad skill
    plt.figure()

    basic_run_split = min([i for _, i in true_asa_runs])
    plot_range(f_itr_to(  basic_runs_data, basic_run_split), color='#7F0000')
    plot_range(f_itr_from(basic_runs_data, basic_run_split), color='k', label='base run')

    plot_range(f_skill_name(true_asa_runs_data, 'Top'), color='r', label='with ASA skill')
    plot_range(f_skill_name(true_asa_runs_data, ideal_bad_skill_names[0]), color='#009F00', label='with ideal skill')
    plot_range(f_skill_name(true_asa_runs_data, ideal_bad_skill_names[1]), color='#FF8C00', label='with bad skill')

    # TODO: NPO
    # plot_range(append_prev_itr(f_exp_name(resumed_from_data[11], 'From_all_manual_pnl005_disc09_'), basic_runs_data),
    #            color='r', label='with ASA (TRPO)')
    # plot_range(append_prev_itr(f_exp_name(resumed_from_data[11], 'From_i11_manual_pnl005_disc09_PPO'), basic_runs_data),
    #            color='darkorange', label='with ASA (NPG)')

    if env == 'minibot':
        tidy_plot(y_label=False)
    else:
        tidy_plot()
    text_to_plot('(Coin-gatherer)' if env == 'gridworld' else '(Maze-bot)', fontsize=14)
    show_save_plot('asa-with-ideal-bad')
    if do_legends:
        plt.figure()
        plt.gcf().set_size_inches(w=10, h=10)
        plt.plot(0, 0, color='k', label='Base run')
        plt.plot(0, 0, color='r', label='With ASA skill')
        plt.plot(0, 0, color='#009F00', label='With ideal skill')
        plt.plot(0, 0, color='#FF8C00', label='With bad skill')
        plt.legend(bbox_to_anchor=(1, 1.1), ncol=5)
        show_save_plot('asa-with-ideal-bad-legend')


def plot_manually_triggered():
    # Plot other resumed-from runs
    plt.figure()

    plt.plot(*get_reward_mean(basic_runs_data), color='k', label='base run')

    colors = ['#00CF00', '#00BF00', '#00AF00', '#009F00', '#008F00', '#007F00', '#006F00']
    for i, run in enumerate(other_asa_runs_data):
        plt.plot(*get_reward_mean(run), color=colors[i])
    label = 'manually triggered ASA' if skill_label == 'ASA' else f'manually added {skill_label} skill'
    plt.plot(-100, -100, color='#009F00', label=label)

    tidy_plot(y_label=None)
    text_to_plot('(b)')
    # plt.legend(loc='upper right' if skill_label == 'bad' else 'lower right')
    show_save_plot(f'manually-triggered-{skill_label}')


def plotC(itr=11, text=None):
    # plot integrators
    plt.figure()

    plt.plot(*get_reward_mean(f_itr_to(basic_runs_data, itr)), color='#000000')
    plt.plot(*get_reward_mean(f_itr_from(basic_runs_data, itr)), color='k')

    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'rnd'), basic_runs_data)), color='red')
    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'rndBias'), basic_runs_data)), color='darkorange')

    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'sbptAvg'), basic_runs_data)), color='tab:cyan', linewidth=0.8, alpha=0.8)
    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'sbptFrst'), basic_runs_data)), color='purple', linewidth=0.8, alpha=0.8)
    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'sbptSmthAvg'), basic_runs_data)), color='blue', linewidth=0.8, alpha=0.8)
    plt.plot(*get_reward_mean(append_prev_itr(f_integrator(resumed_from_data[itr], 'startObsAvg'), basic_runs_data)), color='green', linewidth=0.8, alpha=0.8)

    if text:
        plt.text(3, -0.3, text, fontsize=16)

    tidy_plot(y_label=None)
    show_save_plot('plotC')


def plotClegend():
    plt.figure()
    plt.gcf().set_size_inches(w=4, h=3)

    plt.plot(0,0, color='k', label='Base run')

    plt.plot(0,0, color='k', label='Uninformed schemes:')
    plt.plot(0,0, color='red', label='1')
    plt.plot(0,0, color='orange', label='2')

    plt.plot(0,0, color='k', label='Informed schemes:')
    plt.plot(0,0, color='blue', label='3')
    plt.plot(0,0, color='green', label='4')
    plt.plot(0,0, color='tab:cyan', label='5')
    plt.plot(0,0, color='purple', label='6')

    plt.legend()
    show_save_plot('plotC_legend')





if __name__ == '__main__':
    # plot_asa_with_hippo()
    plot_asa_with_ideal_bad_skills()
    # plot_manually_triggered()
    # plotC(3, '(a)')
    # plotC(11, '(b)')
    # plotC(39, '(c)')
    # plotClegend()
