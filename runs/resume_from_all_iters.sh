#!/bin/bash
# Launch asa_resume_with_new_skill.py for all itr_*.pkl files in given folder.
#
# shellcheck disable=SC2045

max_parallel=8

# Check arguments
if [ $# -ne 1 ]; then
  printf "usage: %s <data_folder>\n" $0
  exit 1
fi

# Make tmp dir
tmp_dir=$(date '+resumed_trainings_output-%Y_%m_%d-%H_%M')
mkdir $tmp_dir

# Launch all trainings
i=0
for f in $(ls -tr "$1"/itr_*.pkl); do
  # Wait for batch if we reached max processes
  if [ $i -ge $max_parallel ]; then
    sleep 1
    echo "Waiting for processes..."
    for p in ${back_pids[*]}; do
      wait $p
    done
    unset back_pids
    i=0
  fi

  # Launch another training in batch
  (
    base=$(basename $f)
    out="${tmp_dir}/${base}_out.txt"
    printf "Launching resumed training after %s\n" $base
    ~holas3/garage/sandbox/asa/runs/asa_resume_with_new_skill.py --snapshot $f &> $out && rm $out
    printf "Training from %s finished\n" $base
  ) &
  back_pids[$i]=$!
  i=$((i+1))

done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

# Remove tmp dir if empty
rmdir $tmp_dir

echo "==== ALL DONE ===="
