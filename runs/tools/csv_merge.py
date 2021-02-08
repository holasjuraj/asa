import os
import pandas as pd
from pandas.errors import EmptyDataError
import tensorflow as tf
import dill
import numpy as np

## Merge progress.csv files from multiple experiments into single csv/Excl file


# Define input/output
data_dir = '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Resumed_with_GWTarget_skill'
output_format = 'xlsx'  # 'csv' or xlsx
output_filename = os.path.join(data_dir, 'All_data--' + os.path.basename(data_dir) + '.' + output_format)
output_columns = [
        'ExpName', 'ExpSkill', 'ExpIntegrator', 'ExpResumedFrom', 'ExpSeed', 'ExpDatetime',
        'Iteration', 'AverageDiscountedReturn', 'AverageReturn',
        'StdReturn', 'MaxReturn', 'MinReturn',
        'NumTrajs', 'SuccessfulTrajs', 'ItrTime', 'Time'
]
discrete_actions = 18
output_columns += [f'DiscreteActions/{i}' for i in range(discrete_actions)]


# Prepare TF config
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# Process experiments
data_dfs = []
with tf.Session(config=config) as session:
    for exp_dir in sorted(os.scandir(data_dir), key=lambda d: d.name):
        if not exp_dir.is_dir():
            continue
        print('Processing {}'.format(exp_dir.name))

        # Retrieve experiment info
        exp_info = exp_dir.name.split('--')
        exp_datetime = exp_info[0]
        exp_seed = exp_info[-1]
        exp_resumed_from = exp_integrator = exp_skill = '-'
        if len(exp_info) == 3:  # Basic run
            exp_name = exp_info[1]
        elif len(exp_info) == 6:  # Resumed from
            r, exp_name, s, i = exp_info[1:5]
            exp_resumed_from = int(r[r.rfind('_')+1 :])
            exp_skill = s[s.find('_')+1 :]
            exp_integrator = i[i.find('_')+1 :]
        else:
            raise Exception('Unable to parse experiment name!')

        # Read progress.csv to get most of the values
        try:
            data = pd.read_csv(os.path.join(exp_dir.path, 'progress.csv'))
        except EmptyDataError:
            continue

        # Read tfevents file to get DiscreteActions/* values
        tfevents_file = [f for f in os.scandir(exp_dir) if f.name.startswith('events.out.tfevents')][0]
        tfevents = tf.train.summary_iterator(tfevents_file.path)
        actions_events = []
        for event in tfevents:
            itr = -1
            acts = np.zeros(discrete_actions)
            for v in event.summary.value:
                if v.tag == 'Iteration':
                    itr = v.simple_value
                elif v.tag.startswith('DiscreteActions'):
                    acts[int(v.tag[-2:-1])] = v.simple_value
            if itr != -1:
                actions_events.append((itr, acts))
        actions_events.sort(key=lambda x: x[0])
        discrete_actions_values = np.array([acts for i, acts in actions_events]).T

        for i, actions_vals in enumerate(discrete_actions_values):
            data[f'DiscreteActions/{i}'] = actions_vals


        # Add experiment info
        data['ExpName'] = exp_name
        data['ExpSkill'] = exp_skill
        data['ExpIntegrator'] = exp_integrator
        data['ExpResumedFrom'] = exp_resumed_from
        data['ExpSeed'] = exp_seed
        data['ExpDatetime'] = pd.to_datetime(exp_datetime, format='%Y_%m_%d-%H_%M')
        data_dfs.append(data)

# Concat all experiments
print('Merging...')
all_data = pd.concat(data_dfs, ignore_index=True, sort=False)
cols = data_dfs[0].columns.tolist()
cols = cols[-4:] + cols[:-4]
all_data = all_data[cols]
print('Total {} records'.format(all_data.shape[0]))

# Write output file
print('Writing output file...')
if output_format == 'csv':
    all_data.to_csv(
            output_filename,
            columns=output_columns
    )
elif output_format == 'xlsx':
    all_data.to_excel(
            output_filename,
            sheet_name='Experiment data',
            index=False,
            columns=output_columns,
            float_format='%.4f',
            freeze_panes=(1, 5)
    )
else:
    raise NotImplementedError

print('Done')
