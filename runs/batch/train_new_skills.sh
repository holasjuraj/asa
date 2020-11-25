#!/bin/bash
# Launch asa_train_new_skill.py for all itr_*.pkl files for all seeds in given folder.
#
# After running Basic_run with multiple seeds, results are stored in subfolders "Basic_run_ABC--sX" for X in [1..N].
# This script launches asa_train_new_skill.py for all N folders = seeds and for all itr_N.pkl files in them.
# However, some runs may fail before they finish (produce final.pkl). This script also looks for such runs, archives their directories and rerun them.

max_parallel=20
usage="
Usage: $(basename $0) -d <data_dir> -n <skill_name> [-g gap] [-i min_itr_num] [-I max_itr_num]
Fix launching asa_train_new_skill.py for all itr_*.pkl files for all seeds in given folder - rerun failed runs.
- all seeds (subfolders of data_dir folder)
- all iterations (itr_N.pkl files)

Options:
  -d data_dir              specify directory with itr_*.pkl files
  -n skill_name            (part of) skill name, used to check for failed runs
  -g gap                   do not run for every itr_*.pkl file, but for every N-th file. Default = 1
  -i iteration_number      minimal iteration number to run for
  -I iteration_number      maximal iteration number to run for
"

# Check arguments
gap=1
data_dir=""
skill_name=""
min_itr_num=0
max_itr_num=9999
while getopts d:p:n:g:i:I: option; do
  case "${option}" in
    d) data_dir=${OPTARG}; ;;
    n) skill_name=${OPTARG}; ;;
    g) gap=${OPTARG}; ;;
    i) min_itr_num=${OPTARG}; ;;
    I) max_itr_num=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ] || [ ! $skill_name ]; then
  echo "$usage"
  exit 1
fi


# Make tmp and "failed" dirs
tmp_dir=$(date '+train_new_skills_trainings-%Y_%m_%d-%H_%M_%S')
mkdir $tmp_dir
script="${tmp_dir}/asa_train_new_skill.py"
cp /home/h/holas3/garage/sandbox/asa/runs/asa_train_new_skill.py $script
experiments_dir="/home/h/holas3/garage/data/local/asa-train-new-skill"
failed_dir="$experiments_dir/../failed/$(basename $experiments_dir)"
mkdir -p $failed_dir

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

    # Do not start training from itr files outside of range
    itr_id=$(basename $itr_f .pkl)  # "itr_N"
    itr_num=${itr_id:4}
    if [[ $itr_num -lt $min_itr_num ]] || [[ $max_itr_num -lt $itr_num ]]; then
      continue
    fi

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

    # Find currect skill policy file
    itr_id=$(basename $itr_f .pkl)  # "itr_N"
    skill_policy_dir=$(ls -d "$experiments_dir"/*after_*"$itr_id"--*"$skill_name"*--s"$seed" 2>/dev/null | tail -1)
    skill_policy_file="$skill_policy_dir/final.pkl"
    if [ $skill_policy_dir ] && [ -f "$skill_policy_file" ]; then
      # Skill policy is ok
      continue
    fi
    if [ $skill_policy_dir ]; then
      # Archive failed run
      echo "Archiving failed run $(basename $skill_policy_dir)"
      mv $skill_policy_dir "$failed_dir/"
    fi

    # Launch another training in batch
    (
      out="${tmp_dir}/${itr_id}_s${seed}_out.txt"
      printf "Launching new skill training from seed %s, %s\n" $seed $itr_id
      # Run
      $script --file $itr_f --seed $seed &> $out  # && rm $out
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

## Remove tmp dir if empty
#rm $script
#rmdir $tmp_dir

echo "==== ALL DONE ===="