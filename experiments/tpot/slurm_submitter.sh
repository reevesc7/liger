#!/bin/bash

script="slurm_runner.sb"

# PARSE COMMAND-LINE ARGUMENTS
while getopts "c:s:e:i:n:" opt; do
    case $opt in
        c)
            config="$OPTARG"
            ;;
        s)
            start_tpotrs="$OPTARG"
            ;;
        e)
            end_tpotrs="$OPTARG"
            ;;
        i)
            id="$OPTARG"
            ;;
        n)
            njobs="$OPTARG"
            ;;
        p)
            pipeparam=$OPTARG
            ;;
        t)
            tpotparam=$OPTARG
            ;;
      *)
            echo "Usage: $0
    -c <config file>
    -r <TPOT random state>
    -s <start TPOT random state>
    -e <end TPOT random state
    -i <run id>
    -n <n jobs>
    -p <pipeline parameters>
    -t <TPOT parameters>"
            exit 1
            ;;
    esac
done

if [[ -z $config ]] && [[ -z $id ]]; then
    echo "No config file or ID set, can't start new run or find checkpoint"
    exit 1
fi

if [[ -n $start_tpotrs && -n $end_tpotrs ]]; then
    for (( tpotrs=$start_tpotrs; tpotrs<$end_tpotrs; tpotrs++ )) do
        echo "SUBMITTING PIPELINE WITH TPOTRS = $tpotrs"
        sbatch --export=CONFIG=$config,TPOTRS=$tpotrs,ID=$id,PIPEPARAM=$pipeparam,TPOTPARAM=$tpotparam,SCRIPT=$script "$script"
    done
elif [[ -n $njobs ]]; then
    for (( n=0; n<$njobs; n++ )) do
        sbatch --export=CONFIG=$config,ID=$id,PIPEPARAM=$pipeparam,TPOTPARAM=$tpotparam,SCRIPT=$script "$script"
    done
else
    echo "Either -s <start TPOT random state> AND -e <end TPOT random state> must be set OR -n <n jobs> must be set"
    exit 1
fi

