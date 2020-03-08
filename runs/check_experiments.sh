#!/bin/bash
# Check number of experiment folders in given path, and their success rate

usage="
Usage: $(basename $0) -d <data_dir> -f <final_file> [-n name] [-x | -o <output_file>]
Check number of experiment folders in given path, and their success rate

Options:
  -d data_dir       specify directory with experiment subdirectories
  -f final_file     name of last generated file - marking successful experiment
  -n name           consider only datadir/*name* folders
  -x                do not produce output file
  -o output_file    name of output file, default = experiments.txt
"

# Check arguments
data_dir=""
final_file=""
name=""
do_output=true
output_file="experiments.txt"
while getopts d:f:n:xo: option; do
  case "${option}" in
    d) data_dir=${OPTARG}; ;;
    f) final_file=${OPTARG}; ;;
    n) name=${OPTARG}; ;;
    x) do_output=false; ;;
    o) output_file=${OPTARG}; ;;
    *) echo "$usage" ; exit 1; ;;
  esac
done

if [ ! $data_dir ] || [ ! $final_file ]; then
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
    echo "" >> $output_file
  fi
done

# Print results
echo "Total:      $total"
echo "Successful: $((total-failed))"
echo "Failed:     $failed"
