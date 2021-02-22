#!/bin/bash
# Launch asa_resume_with_new_skill.py for all possible options:
# - all seeds (subfolders of data_dir folder)
# - all iterations (itr_N.pkl files)
# - all integration methods
#
# shellcheck disable=SC2045

usage="
Usage: $(basename $0) -d <data_dir> (-p <skill_policies_dir> | -P) [-N <exp_name>] [-n <skill_name>] [-g gap] [-i min_itr_num] [-I max_itr_num] [-X max_parallel]
Launch asa_resume_with_new_skill.py for all possible options:
- all seeds (subfolders of data_dir folder)
- all iterations (itr_N.pkl files)
- all integration methods

Options:
  -d data_dir              specify directory with subdirectories of Basic runs
  -p skill_policies_dir    specify directory with subdirectories of skills
  -P                       manually specified skill policy, ignore skill files
  -N exp_name              (part of) experiment name, used to check for failed runs
  -n skill_name            look for skills only in skill_policies_dir/*skill_name* folders
  -g gap                   do not run for every itr_*.pkl file, but for every N-th file. Default = 1
  -i iteration_number      minimal iteration number to run for
  -I iteration_number      maximal iteration number to run for
  -X max_parallel          maximal number of parallel runs. Default = 16
"

# Check arguments
seed="keep"
gap=1
data_dir=""
skill_policies_dir=""
exp_name=""
skill_name=""
ignore_skill_files=false
min_itr_num=0
max_itr_num=9999
max_parallel=16
while getopts d:p:PN:n:g:i:I:X: option; do
  case "${option}" in
    d) data_dir=${OPTARG}; ;;
    p) skill_policies_dir=${OPTARG}; ;;
    P) ignore_skill_files=true; ;;
    N) exp_name=${OPTARG}; ;;
    n) skill_name=${OPTARG}; ;;
    g) gap=${OPTARG}; ;;
    i) min_itr_num=${OPTARG}; ;;
    I) max_itr_num=${OPTARG}; ;;
    X) max_parallel=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ] || [ ! $skill_policies_dir ] && ! $ignore_skill_files; then
  echo "$usage"
  exit 1
fi


# Make tmp dir
tmp_dir="_experiment_outputs/asa_resumed_trainings_output-$exp_name-$(date '+%Y_%m_%d-%H_%M_%S')"
mkdir $tmp_dir
script="${tmp_dir}/asa_resume_with_new_skill.py"
cp /home/h/holas3/garage/sandbox/asa/runs/gridworld/asa_resume_with_new_skill.py $script
experiments_dir="/home/h/holas3/garage/data/local/asa-resume-with-new-skill"
failed_dir="$experiments_dir/../failed/$(basename $experiments_dir)"
mkdir -p $failed_dir

# Launch all trainings
num_pids=0

#for integ_method in $(seq 0 5); do
for integ_method in 3; do  # DEBUG only use specific integrator (3 = SUBPATH_SKILLS_AVG)

  for seed_dir in $(ls -d "$data_dir/"*Basic_run*); do
    seed=`echo "$seed_dir" | sed -n "s/^.*--s\([0-9]\+\).*$/\1/p"`

    # Get number of itr_N.pkl files
    n_itrs=$(ls -1 "$seed_dir/"itr_*.pkl | wc -l)

    itr_gap_i=0
    for itr_i in $(seq 0 $((n_itrs-1)) ); do
      # Skip $gap iterations
      itr_gap_i=$((itr_gap_i+1))
      if [ $itr_gap_i -ne $gap ]; then
        continue
      fi
      itr_gap_i=0

      # Do not start training from last itr_N.pkl
      if [ $itr_i -eq $(($n_itrs-1)) ]; then
        continue
      fi

      # Do not start training from itr files outside of range
      if [[ $itr_i -lt $min_itr_num ]] || [[ $max_itr_num -lt $itr_i ]]; then
        continue
      fi

      # Get itr_N.pkl file
      itr_f="$seed_dir/itr_${itr_i}.pkl"
      itr_id="itr_${itr_i}"  # "itr_N"

      # Wait for batch if we reached max processes
      if [ $num_pids -ge $max_parallel ]; then
        sleep 1
        echo "Waiting for processes..."
        for p in ${back_pids[*]}; do
          wait $p
        done
        unset back_pids
        num_pids=0

        # Clear all but last snapshot files
        rm -f $(for exp_dir in $experiments_dir/*; do ls -tr1 $exp_dir/itr* 2>/dev/null | head -n -1; done)
      fi

      # Get current skill policy file
      if $ignore_skill_files; then
        skill_policy_dir=$skill_name
        skill_policy_file="nofile"
      else
        skill_policy_dir=$(ls -d $skill_policies_dir/*after_*$itr_id--*$skill_name*--s$seed 2>/dev/null | tail -1)
        skill_policy_file="$skill_policy_dir/final.pkl"
        if [ ! $skill_policy_dir ]; then
          echo "Warning: Directory with skill does not exist! Skipping method:$integ_method seed:$seed iteration:$itr_id"
          continue
        fi
        if [ ! -f "$skill_policy_file" ]; then
          echo "Warning: File $skill_policy_file does not exist! Skipping method:$integ_method seed:$seed iteration:$itr_id"
          continue
        fi
      fi

      # Check whether this run was already executed
      exp_dir=$(ls -d $experiments_dir/*resumed_$itr_id--*$exp_name*-skill*$skill_name*-integ$integ_method*--s$seed 2>/dev/null | tail -1)
      exp_done_file="$exp_dir/final.pkl"
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
        out="${tmp_dir}/integ${integ_method}_${itr_id}_s${seed}_out.txt"
        printf "%s    Launching training with integrator %s, seed %s, resumed after %s, with skill %s\n" "$(date +'%x %T')" $integ_method $seed $itr_id "$(basename $skill_policy_dir)"
        # Run
        $script --file $itr_f --skill-policy $skill_policy_file --integration-method $integ_method --seed $seed &> $out  # && rm $out
        printf "%s    Training with integrator %s, seed %s, resumed after %s finished\n" "$(date +'%x %T')" $integ_method $seed $itr_id
      ) &
      back_pids[$num_pids]=$!
      num_pids=$((num_pids+1))
    done
  done
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
