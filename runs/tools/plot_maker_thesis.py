import pandas as pd
import numpy as np
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
            'mb random': 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST07_Resumed_80itrs_discount0.9_pnl0.05\\All_data--Resumed_with_Random_skill.xlsx',
            'mb hippo':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS02_Minibot\\All_data--local.xlsx',
            'mb asa partial match': 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST23_Resumed_Partial_match_Top_skill\\All_data--Resumed_with_Top_Euclidean.xlsx',
            'gw asa':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_Top_skill.xlsx',
            'gw ideal':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWTarget_skill.xlsx',
            'gw bad':    'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_saturn\\data\\archive\\TEST20_Resumed_from_all\\All_data--Resumed_with_GWRandom_skill.xlsx',
            'gw hippo':  'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Code_cogsci_hippo\\rllab-finetuning\\data\\archive\\RUNS03_Gridworld\\All_data--RUNS03_Gridworld.xlsx',
        }
        self.data = DataManager(datafiles)
        self.out_folder = 'C:\\Users\\Juraj\\Documents\\Škola\\FMFI\\PhD\\Dizertačná práca\\img\\plots'
        self.out_folder = os.path.join(self.out_folder, datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        os.mkdir(self.out_folder)

        # General settings
        self.save_plots = False

        # Per-environment settings
        self.env_params = {
            'mb': {  # Minibot
                'high_low_percentile': 5,
                'unwanted_seeds': [],
                'true_asa_runs' : [  # (seed, resumed_from_data) tuples
                    (1, 11), (2, 11), (3, 11), (4, 11), (5, 15), (6, 11), (7, 15), (8, 11)
                ],
                'other_asa_runs' : [  # list of resumed_from_data
                    3, 7, 11, 15, 23, 31, 39
                ]
            },
            'gw': {  # Gridworld
                'high_low_percentile': 25,
                'unwanted_seeds': [8, 16],
                'true_asa_runs' : [  # (seed, resumed_from_data) tuples
                    (12, 109), (13, 49), (14, 59), (15, 49), (3, 49), (4, 69), (5, 49), (7, 79)
                ],
                'other_asa_runs' : [  # list of resumed_from_data
                    19, 49, 79, 109, 139
                ]
            },
        }


    def do_plots(self):
        pass



class DataManager:
    def __init__(self, datafiles):
        """
        Initialize data manager with given data files
        :param datafiles: dict name_of_dataset -> path_to_file
        """
        self.datafiles = datafiles
        self.datasets = dict()

    def ensure_dataset_loaded(self, name):
        """
        Load dataset into memory, if not already done
        """
        if name not in self.datasets:
            pd_data = pd.read_excel(name)
            data = pd.DataFrame.to_dict(pd_data, 'records')
            self.datasets[name] = data

    def builder(self, for_dataset):
        self.ensure_dataset_loaded(for_dataset)
        return DatasetBuilder(self.datasets[for_dataset])

    def __call__(self, for_dataset):
        return self.builder(for_dataset)



class DatasetBuilder:
    def __init__(self, data):
        self.orig_data = data
        self.data = list(data)


    def get_all(self):
        """
        Terminal operation: get a working copy of whole dataset (list of dicts)
        """
        return list(self.data)

    def get_aggr(self, attribute, aggregator=None, **kwargs):
        """
        Terminal operation: get aggregation from desired attribute. Default aggregation is mean.
        Returns tuple (iterations, values)
        """
        if aggregator is None:
            aggregator = np.mean
        data = self.get_all()
        itrs = {row['Iteration'] for row in data}
        itrs = sorted(list(itrs))
        vals = []
        for itr in itrs:
            itr_data = DatasetBuilder(self.orig_data).filter_itr(itr).get_all()
            vals.append(
                aggregator([row[attribute] for row in itr_data], **kwargs)
            )
        return itrs, vals

    def get_reward_mean(self, **kwargs):
        return self.get_aggr('AverageDiscountedReturn', **kwargs)

    def get_reward_percentile(self, q, **kwargs):
        return self.get_aggr('AverageDiscountedReturn', np.percentile, q=q, **kwargs)


    def append_prev_itr(self):
        """
        Append data of previous iteration from Basic runs to dataset
        """
        self.data = self.get_all()  # need to materialize, since filter can be consumed only once
        min_itr = min({row['Iteration'] for row in self.data})
        basic_runs = DatasetBuilder(self.orig_data)\
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

    def filter_exp_name(self, exp_name):
        self.data = filter(
            lambda row: row['ExpName'] == exp_name,
            self.data)
        return self

    def filter_skill_name(self, skill_name, this_or_others=True):
        # Return this skill if this_or_others==True, or all other skills if this_or_others=False
        self.data = filter(
            lambda row: (row['ExpSkill'] == skill_name) == this_or_others,
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

    def filter_unwanted_seeds(self, unwanted_seeds, env=None):
        # Use as `filter_unwanted_seeds([1,2,3])`  or  `filter_unwanted_seeds({'mb': {'unwanted_seeds': [1,2,3]}}, 'mb')`
        if type(unwanted_seeds) is dict:
            unwanted_seeds = unwanted_seeds[env]['unwanted_seeds']
        def filter_func(row):
            s = row['ExpSeed']
            if type(s) is str:
                return int(s[1:]) not in unwanted_seeds
            else:
                return s not in unwanted_seeds
        self.data = filter(filter_func, self.data)
        return self

    def filter_seed_and_resumed_from(self, seed, itr):
        self.data = filter(
            lambda row: row['ExpSeed'] == f's{seed}'  and  row['ExpResumedFrom'] == itr,
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
