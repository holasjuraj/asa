#!/bin/bash
# Launch asa_basic_run.py for all seeds.

max_parallel=8
max_seed=8
usage="
Usage: $(basename $0) [-N <exp_name> [-f <final_file>]]
Launch asa_basic_run.py for all seeds

Options:
  -N exp_name      (part of) experiment name, used to check for failed runs
  -f final_file    name of final file marking successful run. Default = final.pkl
"

# Check arguments
exp_name="NOTHING"
final_file="final.pkl"
while getopts N:f: option; do
  case "${option}" in
    N) exp_name=${OPTARG}; ;;
    f) final_file=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

# Make tmp dir
tmp_dir=$(date '+asa_basic_run_output-%Y_%m_%d-%H_%M')
mkdir $tmp_dir
script="${tmp_dir}/asa_basic_run.py"
cp /home/h/holas3/garage/sandbox/asa/runs/gridworld/asa_basic_run.py $script
experiments_dir="/home/h/holas3/garage/data/local/asa-basic-run"
failed_dir="$experiments_dir/../failed/$(basename $experiments_dir)"
mkdir -p $failed_dir

# Launch all trainings
num_pids=0
for seed in $(seq 1 $max_seed); do
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

  # Check whether this run was already executed
  exp_dir=$(ls -d $experiments_dir/*$exp_name*--s$seed 2>/dev/null | tail -1)
  exp_done_file="$exp_dir/$final_file"
  if [ $exp_dir ] && [ -f "$exp_done_file" ]; then
    # This experiment was successfully executed
    continue
  fi
  if [ $exp_dir ]; then
    # Archive failed run
    echo "Archiving failed run $(basename $exp_dir)"
    mv $exp_dir "$failed_dir/"
  fi

  # Launch another training in batch
  (
    out="${tmp_dir}/${seed}_out.txt"
    printf "Launching training for seed %s\n" $seed
    $script --seed $seed &> $out && rm $out
    printf "Training for seed %s finished\n" $seed
  ) &
  back_pids[$num_pids]=$!
  num_pids=$((num_pids+1))
  seed=$((seed+1))
done

# Wait for last batch
sleep 1
echo "Waiting for processes..."
for p in ${back_pids[*]}; do
  wait $p
done

## Remove tmp dir if empty
#rm $script
#rmdir $tmp_dir

echo "==== ALL DONE ===="

