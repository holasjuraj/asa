#!/bin/bash
# Little dirty script to launch asa_resume_with_new_skill.py for all itr_*.pkl files in all seed-specific folders.
#
# After running Basic_run with multiple seeds, results are stored in subfolders "Basic_run_ABC--sX" for X in [1..N].
# This script launches resume_from_all_iters.sh for all N folders = seeds.

exp_name="Basic_run_20itrs_mapsAll_b15000"
gap=2

seed=1
#for dir in ../../data/local/asa-basic-run/*$exp_name*; do
for dir in ../../data/archive/From_all_with_MinibotRight_mapsAll_b15000/*$exp_name*; do
    runs/resume_from_all_iters.sh -s $seed -g $gap -d $dir
    seed=$((seed+1))
done
