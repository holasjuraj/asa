#!/bin/bash
# Launch asa_resume_with_new_skill.py for all possible options:
# - all seeds (subfolders of data_dir folder)
# - all iterations (itr_N.pkl files)
# - all integration methods
#
# shellcheck disable=SC2045

max_parallel=8
usage="
Usage: $(basename $0) [-g gap] -d <data_dir> -p <skill_policies_dir> [-n <skill_name>]
Launch asa_resume_with_new_skill.py for all possible options:
- all seeds (subfolders of data_dir folder)
- all iterations (itr_N.pkl files)
- all integration methods

Options:
  -g gap                   do not run for every itr_*.pkl file, but for every N-th file. Default = 1
  -d data_dir              specify directory with itr_*.pkl files
  -p skill_policies_dir    specify directory with subdirectories of skills
  -n skill_name            look for skills only in skill_policies_dir/*skill_name* folders
"

# Check arguments
seed="keep"
gap=1
data_dir=""
skill_policies_dir=""
skill_name=""
while getopts g:d:p:n: option; do
  case "${option}" in
    g) gap=${OPTARG}; ;;
    d) data_dir=${OPTARG}; ;;
    p) skill_policies_dir=${OPTARG}; ;;
    n) skill_name=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ] || [ ! $skill_policies_dir ]; then
  echo "$usage"
  exit 1
fi


# Make tmp dir
tmp_dir=$(date '+resumed_trainings_output-%Y_%m_%d-%H_%M_%S')
mkdir $tmp_dir
script="${tmp_dir}/asa_resume_with_new_skill.py"
cp /home/h/holas3/garage/sandbox/asa/runs/asa_resume_with_new_skill.py $script

# Launch all trainings
num_pids=0

for integ_method in $(seq 0 5); do

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

      # Get currect skill policy file
      itr_id=$(basename $itr_f .pkl)  # "itr_N"
      skill_exists=true
      skill_policy_dir=$(ls -d "$skill_policies_dir"/*after_*"$itr_id"--*"$skill_name"*--s"$seed" 2>/dev/null)  || skill_exists=false
      skill_policy_file="$skill_policy_dir/final.pkl"
      if ! $skill_exists; then
        echo "Directory with skill does not exist! Skipping method:$integ_method seed:$seed iteration:$itr_id"
        continue
      fi
      if [ ! -f "$skill_policy_file" ]; then
        echo "File $skill_policy_file does not exist! Skipping method:$integ_method seed:$seed iteration:$itr_id"
        continue
      fi

      # Launch another training in batch
      (
        out="${tmp_dir}/integ${integ_method}_${itr_id}_s${seed}_out.txt"
        printf "Launching training with integrator %s, seed %s, resumed after %s, with skill %s\n" $integ_method $seed $itr_id "$(basename $skill_policy_dir)"
        # Run
        $script --file $itr_f --skill-policy $skill_policy_file --integration-method $integ_method --seed $seed &> $out && rm $out
        printf "Training with integrator %s, seed %s, resumed after %s finished\n" $integ_method $seed $itr_id
      ) &
      back_pids[$num_pids]=$!
      num_pids=$((num_pids+1))
    done
  done
done

# Wait for last batch
for p in ${back_pids[*]}; do
  wait $p
done

# Remove tmp dir if empty
rmdir $tmp_dir

echo "==== ALL DONE ===="
