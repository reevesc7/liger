#!/bin/bash

script="slurm_runner.sb"
defaultlist="unfinished.txt"

# PARSE COMMAND-LINE ARGUMENTS
while getopts "l:" opt; do
    case $opt in
        l)
            list="$OPTARG"
            ;;
        *)
            echo "Usage: $0
    -l <list of unfinished runs file>"
            exit 1
            ;;
    esac
done

if [[ -z $list ]]; then
    list=$defaultlist
fi

while IFS= read -r id; do
    sbatch --export=ID=$id,SCRIPT=$script "$script"
done < $list

