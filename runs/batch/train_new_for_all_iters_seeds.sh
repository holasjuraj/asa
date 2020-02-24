#!/bin/bash
# Launch asa_train_new_skill.py for all itr_*.pkl files for all seeds in given folder.
#
# After running Basic_run with multiple seeds, results are stored in subfolders "Basic_run_ABC--sX" for X in [1..N].
# This script launches asa_train_new_skill.py for all N folders = seeds and for all itr_N.pkl files in them.

max_parallel=8
usage="
Usage: $(basename $0) [-g gap] -d <data_dir>
Launch asa_train_new_skill.py for all itr_*.pkl files for all seeds in given folder.

Options:
  -g gap         do not run for every itr_*.pkl file, but for every N-th file. Default = 1
  -d data_dir    specify directory with itr_*.pkl files
"

# Check arguments
gap=1
data_dir=""
while getopts g:d: option; do
  case "${option}" in
    g) gap=${OPTARG}; ;;
    d) data_dir=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ]; then
  echo "$usage"
  exit 1
fi


# Make tmp dir
tmp_dir=$(date '+train_new_skill_trainings-%Y_%m_%d-%H_%M_%S')
mkdir $tmp_dir
script="${tmp_dir}/asa_train_new_skill.py"
cp /home/h/holas3/garage/sandbox/asa/runs/asa_train_new_skill.py $script

# Launch all trainings
num_pids=0
for seed_dir in $(ls -d "$data_dir/"*Basic_run*); do
  seed=`echo "$seed_dir" | sed -n "s/^.*--s\([0-9]\+\).*$/\1/p"`
  itr_i=0
  for itr_f in $(ls -tr "$seed_dir/"itr_*.pkl); do
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
      itr_id=$(basename $itr_f .pkl)  # "itr_N"
      out="${tmp_dir}/${itr_id}_s${seed}_out.txt"
      printf "Launching new skill training from seed %s, %s\n" $seed $itr_id
      # Run
      $script --file $itr_f --seed $seed &> $out && rm $out
      printf "Training from seed %s, %s finished\n" $seed $itr_id
    ) &
    back_pids[$num_pids]=$!
    num_pids=$((num_pids+1))

  done
done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

# Remove tmp dir if empty
rm $script
rmdir $tmp_dir

echo "==== ALL DONE ===="

