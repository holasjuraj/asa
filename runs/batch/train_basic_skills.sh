#!/bin/bash
# Launch asa_train_basic_skill.py for all specified skill IDs.

usage="
Usage: $(basename $0) -n <skill_name> [-X max_parallel]
Launch asa_train_basic_skill.py for all specified skill IDs (also rerun failed runs).

Options:
  -n skill_name            (part of) skill name, used to check for failed runs
  -X max_parallel          maximal number of parallel runs. Default = 16
"

# Check arguments
skill_name=""
max_parallel=16
busy_wait_time=10  # seconds
while getopts n:X: option; do
  case "${option}" in
    n) skill_name=${OPTARG}; ;;
    X) max_parallel=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $skill_name ]; then
  echo "$usage"
  exit 1
fi


# Make tmp and "failed" dirs
tmp_dir="_experiment_outputs/asa_train_basic_skills_outputs-$skill_name-$(date '+%Y_%m_%d-%H_%M_%S')"
mkdir $tmp_dir
script="${tmp_dir}/asa_train_basic_skill.py"
cp /home/h/holas3/garage/sandbox/asa/runs/gridworld/asa_train_basic_skill.py $script
experiments_dir="/home/h/holas3/garage/data/local/asa-train-basic-skill"
failed_dir="$experiments_dir/../failed/$(basename $experiments_dir)"
mkdir -p $failed_dir
echo "Using temporary directory $tmp_dir ."
echo "Create file named 'STOP' in temporary directory to interrupt this script."
printf "\n\n"

# Launch all trainings
num_pids=0
num_launched=0

  for skill_id in $(seq 0 13); do
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
    skill_policy_dir=$(ls -d "$experiments_dir"/*"$skill_name"*-trg"$skill_id" 2>/dev/null | tail -1)
    skill_policy_file="$skill_policy_dir/final.pkl"
    if [ $skill_policy_dir ] && [ -f "$skill_policy_file" ]; then
      # This experiment was successfully executed
      continue
    fi
    if [ $skill_policy_dir ]; then
      # Archive failed run
      echo "Archiving failed run $(basename $skill_policy_dir)"
      mv $skill_policy_dir "$failed_dir/"
    fi

    # Launch another training in batch
    num_launched=$((num_launched+1))
    (
      out="${tmp_dir}/trg${skill_id}_out.txt"
      printf "%s    Launching %s-th new skill training: skill_id %s\n" "$(date +'%F %T')" $num_launched $skill_id
      # Run
      $script --skillid $skill_id &> $out  # && rm $out
      printf "%s    Training for skill_id %s finished\n" "$(date +'%F %T')" $skill_id
    ) &
    back_pids[$num_pids]=$!
    num_pids=$((num_pids+1))
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
