#!/usr/bin/env python

import os
import shutil
import pandas as pd
from pandas.errors import EmptyDataError

## Sort trained skill policies by their (smoothed) score, and separate successful skills from unsuccessful.



# Define input/output
data_dir         = '/home/h/holas3/garage/data/local/asa-train-new-skill'
data_dir   = '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all_Wrong_stop_fun/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.75'
successful_dir   = '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all_Wrong_stop_fun/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.25--bad_a0.75'
unsuccessful_dir = '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all_Wrong_stop_fun/Skill_policies/Skill_Top_T20_sbpt2to4--bad_a0.25'
dry_run          = True  # do not actually move folders

# Define sorting preferences
sort_metric       = 'AverageReturn'
smoothing_factor  = 0.95
success_threshold = 0.25



# Process experiments
successful_skills = []
num_suc = num_unsuc = 0
for exp_dir in sorted(os.scandir(data_dir), key=lambda d: d.name):
    if not exp_dir.is_dir():
        continue
    print('Processing {}: '.format(exp_dir.name), end='')

    # Retrieve experiment info
    _, it, _, exp_seed = exp_dir.name.split('--')
    exp_itr = int(it[it.rfind('_')+1 :])

    # Read progress.csv to get training the values
    try:
        data = pd.read_csv(os.path.join(exp_dir.path, 'progress.csv'))
    except EmptyDataError:
        print('Error: no data found!')
        continue

    # Determine successful/unsuccessful
    scores = data[sort_metric]
    scores = scores.ewm(alpha=1-smoothing_factor, adjust=False).mean()
    final_score = scores.iat[-1]
    success = final_score >= success_threshold
    print('successful' if success else 'unsuccessful')
    if success:
        num_suc += 1
        successful_skills.append((exp_seed, exp_itr, final_score))
    else:
        num_unsuc += 1

    # Move experiment dir to successful/unsuccessful dir
    if not dry_run:
        target_dir = successful_dir if success else unsuccessful_dir
        shutil.move(exp_dir, os.path.join(target_dir, exp_dir.name))


print('Successful skills:')
for seed, itr, final_score in sorted(successful_skills):
    print(f'\t{seed},  itr_{itr}: {final_score:.3f}')


print(f'\nDone. Processed skills:\n\t{num_suc} successful\n\t{num_unsuc} unsuccessful\n\t{num_suc + num_unsuc} total')
