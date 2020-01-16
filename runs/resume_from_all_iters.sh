#!/bin/bash
# Launch asa_resume_with_new_skill.py for all itr_*.pkl files in given folder.
#
# shellcheck disable=SC2045

max_parallel=8
usage="
Usage: $(basename $0) [-g gap] [-s seed] -d <data_dir>
Launch asa_resume_with_new_skill.py for all itr_*.pkl files in given folder.

Options:
  -g gap         do not run for every itr_*.pkl file, but for every N-th file. Default = 1
  -s seed        specify seed for training. Default = \"keep\"
  -d data_dir    specify directory with itr_*.pkl files
"

# Check arguments
seed="keep"
gap=1
data_dir=""
while getopts s:g:d: option; do
  case "${option}" in
    g) gap=${OPTARG}; ;;
    s) seed=${OPTARG}; ;;
    d) data_dir=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ]; then
  echo "$usage"
  exit 1
fi


# Make tmp dir
tmp_dir=$(date '+resumed_trainings_output-%Y_%m_%d-%H_%M')
mkdir $tmp_dir

# Launch all trainings
num_pids=0
itr_i=0
for f in $(ls -tr "$data_dir"/itr_*.pkl); do
  # Skip $gap iterations
  itr_i=$((itr_i+1))
  if [ $itr_i -ne $gap ]; then
    continue
  fi
  itr_i=0

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
    base=$(basename $f)
    out="${tmp_dir}/${base}_out.txt"
    printf "Launching resumed training after %s\n" $base
    ~holas3/garage/sandbox/asa/runs/asa_resume_with_new_skill.py --file $f --seed $seed &> $out && rm $out
    printf "Training from %s finished\n" $base
  ) &
  back_pids[$num_pids]=$!
  num_pids=$((num_pids+1))

done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

# Remove tmp dir if empty
rmdir $tmp_dir

echo "==== ALL DONE ===="
