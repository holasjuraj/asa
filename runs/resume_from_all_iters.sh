#!/bin/bash
# Launch asa_resume_with_new_skill.py for all itr_*.pkl files in given folder.
#
# shellcheck disable=SC2045

if [ $# -ne 1 ]; then
  printf "usage: %s <data_folder>\n" $0
  exit 1
fi

for f in $(ls -tr "$1"/itr_*.pkl); do
  base=$(basename $f)
  printf "################ LAUNCHING RESUMED TRAINING FROM %s ################\n" $base
  ~holas3/garage/sandbox/asa/runs/asa_resume_with_new_skill.py --snapshot $f
  printf "#################### TRAINING FROM %s FINISHED #####################\n\n\n\n" $base

done
