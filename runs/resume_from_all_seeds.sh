#!/bin/bash
# Launch asa_resume_with_new_skill.py from specific iteration for all seeds.
#
# After running Basic_run with multiple seeds, results are stored in subfolders "Basic_run_ABC--sX" for X in [1..N].
# This script resumes training from specified iteration (asa_resume_with_new_skill.sh --file itr_N.pkl) for all N folders = seeds.

itr="1"

max_parallel=8

# Make tmp dir
tmp_dir=$(date '+resumed_trainings_output-%Y_%m_%d-%H_%M')
mkdir $tmp_dir

# Launch all trainings
seed=1
num_pids=0
for dir in ../../data/archive/Skill_integrator_test1_mapsAll_b5000/*Basic_run*; do
  # Wait for batch if we reached max processes
  if [ $num_pids -ge $max_parallel ]; then
    sleep 1
    echo "Waiting for processes..."
    for p in ${back_pids[*]}; do
      wait $p
    done
    unset back_pids
    num_pids=0
  fi

  # Launch another training in batch
  (
    out="${tmp_dir}/${seed}_out.txt"
    f="${dir}/itr_${itr}.pkl"
    printf "Launching resumed training for seed %s\n" $seed
    ~holas3/garage/sandbox/asa/runs/asa_resume_with_new_skill.py --file $f --seed $seed &> $out && rm $out
    printf "Training from %s finished\n" $seed
  ) &
  back_pids[$num_pids]=$!
  num_pids=$((num_pids+1))
  seed=$((seed+1))
done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

# Remove tmp dir if empty
rmdir $tmp_dir

echo "==== ALL DONE ===="

