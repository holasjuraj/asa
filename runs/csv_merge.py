import os
import pandas as pd
from pandas.errors import EmptyDataError

## Merge progress.csv files from multiple experiments into single csv/Excl file


# Define input/output
data_dir = '/home/h/holas3/garage/data/archive/From_all_with_MinibotRight'
output_format = 'xlsx'  # 'csv' or xlsx
output_filename = os.path.join(data_dir, 'all_experiments.' + output_format)
output_columns = ['ExpName', 'ExpResumedFrom', 'ExpSeed', 'ExpDatetime',
                  'Iteration', 'AverageDiscountedReturn', 'AverageReturn',
                  'NumTrajs', 'StdReturn', 'MaxReturn', 'MinReturn', 'Time',
                  'ItrTime']


# Process experiments
data_dfs = []
for exp_dir in sorted(os.scandir(data_dir), key=lambda d: d.name):
    if not exp_dir.is_dir():
        continue
    # Retrieve experiment info
    exp_info = exp_dir.name.split('--')
    exp_datetime, exp_name = exp_info[:2]
    exp_resumed_from = '-'
    if exp_name.startswith('resumed_from_itr_'):
        exp_resumed_from = int(exp_name[exp_name.rfind('_')+1 :])
        exp_name = exp_info[2]
    exp_seed = exp_info[-1]
    # Read progress.csv
    try:
        data = pd.read_csv(os.path.join(exp_dir.path, 'progress.csv'))
    except EmptyDataError:
        continue
    # Add experiment info
    data['ExpName'] = exp_name
    data['ExpResumedFrom'] = exp_resumed_from
    data['ExpSeed'] = exp_seed
    data['ExpDatetime'] = pd.to_datetime(exp_datetime, format='%Y_%m_%d-%H_%M')
    data_dfs.append(data)

# Concat all experiments
all_data = pd.concat(data_dfs, ignore_index=True, sort=False)
cols = data_dfs[0].columns.tolist()
cols = cols[-4:] + cols[:-4]
all_data = all_data[cols]

# Write output file
if output_format == 'csv':
    all_data.to_csv(
            output_filename,
            columns=output_columns
    )
elif output_format == 'xlsx':
    all_data.to_excel(
            output_filename,
            sheet_name=os.path.basename(data_dir),
            index=False,
            columns=output_columns,
            float_format='%.4f',
            freeze_panes=(1, 3)
    )
else:
    raise NotImplementedError
