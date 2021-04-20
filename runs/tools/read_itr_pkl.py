#!/usr/bin/env python

import os
import tensorflow as tf
import dill
import joblib
from garage.misc.tensor_utils import unflatten_tensors
from sandbox.asa.utils.path_trie import PathTrie
from sandbox.asa.envs import GridworldGathererEnv
import matplotlib.pyplot as plt
import numpy as np

# Snippet for starting interactive TF session and reading exported training snapshot

# If GPUs are blocked by another user, force use specific GPU (0 or 1), or run on CPU (-1).

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


seed, itr, file_path = 3, 49, '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.75/2021_02_25-02_53--after_itr_49--Skill_Top_T20_sbpt2to4--s3/final.pkl'
# seed, itr, file_path = 5, 49, '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.75/2021_02_25-16_17--after_itr_49--Skill_Top_T20_sbpt2to4--s5/final.pkl'
# seed, itr, file_path = 5, 119, '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.75/2021_02_25-21_17--after_itr_119--Skill_Top_T20_sbpt2to4--s5/final.pkl'
# seed, itr, file_path = 5, 129, '/home/h/holas3/garage/data/archive/TEST20_Resumed_from_all/Skill_policies/Skill_Top_T20_sbpt2to4--good_a0.75/2021_02_25-12_28--after_itr_129--Skill_Top_T20_sbpt2to4--s5/final.pkl'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
# saved_data = joblib.load(path)
with open(file_path, 'rb') as file:
    saved_data = dill.load(file)

# # Read weights
# old_top_policy = saved_data['policy']
# otp_weight_values = unflatten_tensors(
#     old_top_policy.get_param_values(),
#     old_top_policy.get_param_shapes()
# )
# otp_weights = list(zip(
#     [p.name for p in old_top_policy.get_params()],
#     otp_weight_values
# ))
#
# for name, value in otp_weights:
#     print(name, value.shape)
#     print(value.T)


# # Inspect frequent path
# trie_min_length = 1
# trie_max_length = 1
# trie_action_map = {i: ch for i, ch in enumerate('ABCDEFGHIJKLM^>v<#')}  # for Gridworld 13reg
# trie_min_f_score = 1
# trie_max_results = 7
# trie_min_count = 10
#
# paths = saved_data['paths']
# path_trie = PathTrie(saved_data['hrl_policy'].num_skills)
# for path in paths:
#     actions = path['actions'].argmax(axis=1).tolist()
#     observations = path['observations']
#     path_trie.add_all_subpaths(actions, observations,
#         min_length=trie_min_length, max_length=trie_max_length
#     )
# frequent_paths = path_trie.items(
#     action_map=trie_action_map, min_count=trie_min_count,
#     min_f_score=trie_min_f_score, max_results=trie_max_results, aggregations=[]
# )
#
# # top_path = frequent_paths[0]
# top_path = path_trie.item_for_path([17], action_map=trie_action_map)  # new skill
# start_poss = top_path['end_observations'][:, :2]
#
# GridworldGathererEnv()._plot_visitations([])
# plt.scatter(start_poss[:, 1], 45 - start_poss[:, 0], alpha=0.05)
# plt.savefig(f'ends_Jvvv_itr_{itr}.png')


# Inspect new skill policy file
new_skill_subpath = saved_data['subpath']
start_poss = new_skill_subpath['start_observations'][:, :2]
end_poss = new_skill_subpath['end_observations'][:, :2]

GridworldGathererEnv()._plot_visitations([])
plt.scatter(start_poss[:, 1], 45 - start_poss[:, 0], c='green', alpha=0.25)
plt.scatter(end_poss[:, 1], 45 - end_poss[:, 0], c='red', alpha=0.15)
plt.savefig(f'skill_Top_s{seed}_itr{itr}--start_ends.png', dpi=300)
# plt.savefig(f'plain_map.png', dpi=300)



sess.close()
