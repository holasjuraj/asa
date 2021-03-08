#!/bin/bash
# Launch asa_basic_run.py for all seeds.

usage="
Usage: $(basename $0) [-N <exp_name> [-f <final_file>]] [-S num_seeds] [-X max_parallel]
Launch asa_basic_run.py for all seeds

Options:
  -N exp_name      (part of) experiment name, used to check for failed runs
  -f final_file    name of final file marking successful run. Default = final.pkl
  -S num_seeds     number of seeds to run. Default = 16
  -X max_parallel  maximal number of parallel runs. Default = 16
"

# Check arguments
exp_name="noname"
final_file="final.pkl"
num_seeds=16
max_parallel=16
busy_wait_time=60  # seconds
while getopts N:f:S:X: option; do
  case "${option}" in
    N) exp_name=${OPTARG}; ;;
    f) final_file=${OPTARG}; ;;
    S) num_seeds=${OPTARG}; ;;
    X) max_parallel=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

# Make tmp dir
tmp_dir="_experiment_outputs/asa_basic_run_output-$exp_name-$(date '+%Y_%m_%d-%H_%M_%S')"
mkdir $tmp_dir || exit 1;
script="${tmp_dir}/asa_basic_run.py"
cp /home/h/holas3/garage/sandbox/asa/runs/gridworld/asa_basic_run.py $script
experiments_dir="/home/h/holas3/garage/data/local/asa-basic-run"
failed_dir="$experiments_dir/../failed/$(basename $experiments_dir)"
mkdir -p $failed_dir
echo "Using temporary directory $tmp_dir ."
echo "Create file named 'STOP' in temporary directory to interrupt this script."
printf "\n\n"

# Launch all trainings
num_pids=0
for seed in $(seq 1 $num_seeds); do
  # Break if stop file is found
  if [ -f "$tmp_dir/STOP" ]; then
    echo "Warning: Interrupting script, stop file was found"
    break 10
  fi

  # Wait for processes if we reached max processes
  if [ $num_pids -ge $max_parallel ]; then
    sleep 1
    echo "Waiting for processes..."
    while [ $num_pids -ge $max_parallel ]; do
      # Break if stop file is found
      if [ -f "$tmp_dir/STOP" ]; then
        echo "Warning: Interrupting script, stop file was found"
        break 10
      fi
      # Wait
      sleep $busy_wait_time
      # Check which back_pids are still running
      old_back_pids=("${back_pids[*]}")
      unset back_pids
      num_pids=0
      for p in ${old_back_pids[*]}; do
        if ps -p $p > /dev/null; then
          back_pids[$num_pids]=$p
          num_pids=$((num_pids+1))
        fi
      done
    done
  fi

  # Check whether this run was already executed
  exp_dir=$(ls -d $experiments_dir/*$exp_name*-s$seed 2>/dev/null | tail -1)
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
    printf "%s    Launching training for seed %s\n" "$(date +'%F %T')" $seed
    $script --seed $seed &> $out  # && rm $out
    printf "%s    Training for seed %s finished\n" "$(date +'%F %T')" $seed
  ) &
  back_pids[$num_pids]=$!
  num_pids=$((num_pids+1))
  seed=$((seed+1))
done

# Wait for last batch
sleep 1
echo "Waiting for last processes..."
for p in ${back_pids[*]}; do
  wait $p
done

## Remove tmp dir if empty
#rm $script
#rmdir $tmp_dir

echo "==== ALL DONE ===="

