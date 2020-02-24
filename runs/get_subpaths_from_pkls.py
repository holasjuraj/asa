import os
import tensorflow as tf
import dill
import numpy as np

from sandbox.asa.utils.path_trie import PathTrie

# If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_dir = '/home/h/holas3/garage/data/local/asa-test'
output_filename = os.path.join(data_dir, 'All_subpaths_from_pkls.tsv')
output_file = open(output_filename, 'w')

trie_min_length = 3
trie_max_length = 5
trie_action_map = {0: 's', 1: 'L', 2: 'R'}
trie_min_f_score = 1
trie_max_results = 7
output_file.write('Paths from {}, length {}-{}, {} paths per iteration\n'
                  .format(data_dir, trie_min_length, trie_max_length, trie_max_results))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

for exp_dir in sorted(os.scandir(data_dir), key=lambda d: d.name):
    if not exp_dir.is_dir():
        continue
    print('Processing {}'.format(exp_dir.name))
    # Retrieve experiment info
    exp_info = exp_dir.name.split('--')
    exp_datetime, exp_name, exp_seed = exp_info

    # Read all snapshots
    itr_pkls = [f for f in os.scandir(exp_dir) if f.name.startswith('itr_')]
    itr_pkls.sort(key=lambda f: int(f.name[4:-4]))
    output_values = np.zeros((trie_max_results, len(itr_pkls)*3), dtype='U8')

    for i_itr, itr_pkl in enumerate(itr_pkls):
        print('\t{}'.format(itr_pkl.name))
        # Read itr_N.pkl file
        with tf.Session(config=config) as session:
            with open(itr_pkl.path, 'rb') as itr_file:
                saved_data = dill.load(itr_file)
        tf.reset_default_graph()

        # Build Trie
        paths = saved_data['paths']
        path_trie = PathTrie(saved_data['hrl_policy'].num_skills)
        for path in paths:
            actions = path['actions'].argmax(axis=1).tolist()
            observations = path['observations']
            path_trie.add_all_subpaths(actions, observations,
                min_length=trie_min_length, max_length=trie_max_length
            )
        frequent_paths = path_trie.items(
            action_map=trie_action_map, min_count=10,
            min_f_score=trie_min_f_score, max_results=trie_max_results, aggregations=[]
        )
        # Save output values
        for i_sbpt, subpath in enumerate(frequent_paths):
            output_values[i_sbpt, (i_itr*3):(i_itr*3+3)] = np.asarray([
                subpath['actions_text'],
                '{:.2f}'.format(subpath['f_score']).replace('.', ','),
                '{:d}'.format(subpath['count']),
            ])

    # Dump output values for this experiment to file
    for row in output_values:
        output_file.write('\t'.join(row))
        output_file.write('\n')


output_file.close()
