#!/bin/bash
# Check number of experiment folders in given path, and their success rate

usage="
Usage: $(basename $0) -d <data_dir> [-n name] [-f <final_file>] [-o | -O <output_file>]
Check number of experiment folders in given path, and their success rate

Options:
  -d data_dir       specify directory with experiment subdirectories
  -n name           consider only datadir/*name* folders default = \"\"
  -f final_file     name of last generated file - marking successful experiment, default = final.pkl
  -o                produce output file with default name experiments_status.txt
  -O output_file    produce output file with specified name
"

# Check arguments
data_dir=""
name=""
final_file="final.pkl"
do_output=false
output_file="experiments_status.txt"
while getopts d:n:f:oO: option; do
  case "${option}" in
    d) data_dir=${OPTARG}; ;;
    n) name=${OPTARG}; ;;
    f) final_file=${OPTARG}; ;;
    o) do_output=true; ;;
    O) do_output=true; output_file=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ]; then
  echo "$usage"
  exit 1
fi

# Perform experiments check
total=0
failed=0
if $do_output; then
  printf "" > $output_file
fi

for exp_dir in $data_dir/*$name*; do
  total=$((total+1))
  if $do_output; then
    printf "%s" $(basename $exp_dir) >> $output_file
  fi
  exp_ok=true
  ls $exp_dir/*$final_file* &>/dev/null  || exp_ok=false
  if ! $exp_ok; then
    failed=$((failed+1))
    if $do_output; then
      printf "\tfailed\n" >> $output_file
    fi
  else
    if $do_output; then
      echo "" >> $output_file
    fi
  fi
done

# Print results
echo "Total:      $total"
echo "Successful: $((total-failed))"
echo "Failed:     $failed"
