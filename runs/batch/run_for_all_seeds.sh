#!/bin/bash
# Launch asa_basic_run.py for all seeds.

max_parallel=8
max_seed=8

# Make tmp dir
tmp_dir=$(date '+asa_basic_run_output-%Y_%m_%d-%H_%M')
mkdir $tmp_dir
script="${tmp_dir}/asa_basic_run.py"
cp /home/h/holas3/garage/sandbox/asa/runs/minibot/asa_basic_run.py $script

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

  # Launch another training in batch
  (
    out="${tmp_dir}/${seed}_out.txt"
    printf "Launching training for seed %s\n" $seed
    $script --seed $seed &> $out  # && rm $out
    printf "Training for seed %s finished\n" $seed
  ) &
  back_pids[$num_pids]=$!
  num_pids=$((num_pids+1))
  seed=$((seed+1))
done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

## Remove tmp dir if empty
#rmdir $tmp_dir

echo "==== ALL DONE ===="

