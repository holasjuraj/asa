import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data_file = 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST7_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Top_skill.xlsx'
# data_file = 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST7_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_sLLLs_disc099_skill.xlsx'
data_file = 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST7_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Random_skill.xlsx'

pd_data = pd.read_excel(data_file)
data = pd.DataFrame.to_dict(pd_data, 'records')


## Filters
lfilter = lambda f, d: list(filter(f, d))
f_basic_run     = lambda d: lfilter(lambda r: r['ExpName'].startswith('Basic_run'), d)
f_exp_name      = lambda d, str: lfilter(lambda r: r['ExpName'] == str, d)
f_resumed_from  = lambda d, itr: lfilter(lambda r: r['ExpResumedFrom'] == itr, d)
f_itr           = lambda d, itr: lfilter(lambda r: r['Iteration'] == itr, d)
f_itr_from      = lambda d, itr: lfilter(lambda r: r['Iteration'] >= itr, d)
f_itr_to        = lambda d, itr: lfilter(lambda r: r['Iteration'] <= itr, d)
f_integrator    = lambda d, integ: lfilter(lambda  r: r['ExpIntegrator'] == integ, d)


## Aggregators
def get_ADR_stat(data, aggr_func, **kwargs):
    itrs = {r['Iteration'] for r in data}
    itrs = sorted(list(itrs))
    vals = []
    for itr in itrs:
        vals.append(aggr_func([r['AverageDiscountedReturn'] for r in f_itr(data, itr)], **kwargs))
    return np.array([itrs, vals])

get_ADR_mean = lambda d: get_ADR_stat(d, np.mean)
get_ADR_low  = lambda d: get_ADR_stat(d, np.percentile, q=2)
get_ADR_high = lambda d: get_ADR_stat(d, np.percentile, q=100-2)


## Helpers
def append_prev_itr(data, basic_run_data):
    min_itr = min({r['Iteration'] for r in data})
    res = list(data)
    res.extend(f_itr(basic_run_data, min_itr-1))
    return res

def plot_range(data, color=None, label=None, alpha=0.2, **kwargs):
    X, M  = get_ADR_mean(data)
    _, Q1 = get_ADR_low(data)
    _, Q3 = get_ADR_high(data)
    plt.plot(X, M, c=color, label=label, **kwargs)
    plt.fill_between(X, Q1, Q3, color=color, alpha=alpha)

def tidy_plot(y_label='Avg. discounted reward', w=4, h=3):
    plt.gcf().set_size_inches(w=w, h=h)
    plt.tight_layout()
    plt.grid()
    plt.xlim(0, 79)
    plt.ylim(-2.5, 0)
    plt.xlabel('Iteration')
    plt.ylabel(y_label)
    if not y_label:
        plt.gca().set_yticklabels([])
        plt.gcf().set_size_inches(w=w-0.5, h=h-0.06)


# Data gathering
basic_run = f_basic_run(data)
res_from = {}
for i in range(3, 40, 4):
    try:
        res_from[i] = append_prev_itr(f_resumed_from(data,  i), basic_run)
    except ValueError:
        pass


## Plotting

def plotA():
    # plot one
    plt.figure()

    plot_range(f_itr_to(basic_run, 11), color='#7F0000')
    plot_range(f_itr_from(basic_run, 11), color='k', label='base run')
    plot_range(res_from[11], color='r', label='with bad skill')

    # plot_range(append_prev_itr(f_exp_name(res_from[11], 'From_all_manual_pnl005_disc09_'), basic_run),
    #            color='r', label='with ASA (TRPO)')
    # # plot_range(append_prev_itr(f_exp_name(res_from[11], 'From_i11_manual_pnl005_disc09_PPO'), basic_run),
    # #            color='purple', label='with ASA (PPO)')
    # plot_range(append_prev_itr(f_exp_name(res_from[11], 'From_i11_manual_pnl005_disc09_PPO'), basic_run),
    #            color='darkorange', label='with ASA (NPG)')

    # plt.text(3, -0.3, '(a)', fontsize=14)
    tidy_plot()
    plt.legend(loc='lower right')
    plt.show()


def plotB():
    # plot many
    plt.figure()

    plt.plot(*get_ADR_mean(f_itr_to(basic_run, 39)), color='#1F5F00')
    plt.plot(*get_ADR_mean(f_itr_from(basic_run, 39)), color='k', label='base run')
    plt.plot(*get_ADR_mean(res_from[11]), color='r', label='with ideal skill')
    plt.plot(*get_ADR_mean(res_from[3]), color='#009F00', label='manually added bad skill')
    for itr, col in zip( [7, 15, 23, 31, 39],
                         ['#006F00',  '#00AF00',   '#007F00',   '#00BF00',   '#008F00']):
    # for itr in [7, 15, 23, 31, 39]:
        plt.plot(*get_ADR_mean(res_from[itr]), color=col)

    # plt.text(3, -0.3, '(b)', fontsize=14)
    # tidy_plot(y_label=None)
    tidy_plot()
    plt.legend(loc='lower right')
    plt.show()


def plotC(itr=11, text=None):
    # plot integrators
    plt.figure()

    plt.plot(*get_ADR_mean(f_itr_to(basic_run, itr)), color='#000000')
    plt.plot(*get_ADR_mean(f_itr_from(basic_run, itr)), color='k')

    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'rnd'), basic_run)), color='red')
    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'rndBias'), basic_run)), color='darkorange')

    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'sbptAvg'), basic_run)), color='tab:cyan', linewidth=0.8, alpha=0.8)
    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'sbptFrst'), basic_run)), color='purple', linewidth=0.8, alpha=0.8)
    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'sbptSmthAvg'), basic_run)), color='blue', linewidth=0.8, alpha=0.8)
    plt.plot(*get_ADR_mean(append_prev_itr(f_integrator(res_from[itr], 'startObsAvg'), basic_run)), color='green', linewidth=0.8, alpha=0.8)

    if text:
        plt.text(3, -0.3, text, fontsize=16)

    tidy_plot(y_label=None)
    plt.show()

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
    plt.show()


if __name__ == '__main__':
    plotA()
    plotB()
    # plotC(3, '(a)')
    # plotC(11, '(b)')
    # plotC(39, '(c)')
    # plotClegend()
