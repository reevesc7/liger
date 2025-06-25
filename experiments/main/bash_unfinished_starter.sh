#!/bin/bash

script="bash_runner.sh"
list="unfinished.txt"

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

while IFS= read -r id; do
    bash $script -i "$id"
done < $list

