#!/bin/bash --login

export PATH=/mnt/ufs18/home-107/jory/projects/chris/py/python/bin:$PATH
export PYTHONPATH=/mnt/ufs18/home-107/jory/projects/chris
export PYTHONHOME=/mnt/ufs18/home-107/jory/projects/chris/py/python

# PARSE COMMAND-LINE ARGUMENTS
while getopts "c:r:i:p:t:" opt; do
    case $opt in
        c)
            config="$OPTARG"
            ;;
        r)
            tpotrs="$OPTARG"
            ;;
        i)
            id="$OPTARG"
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
    -i <run id>
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

all_output=$(lg-tpot \
    --config="$config" \
    --tpotrs="$tpotrs" \
    --id="$id" \
    --pipeparam=$pipeparam \
    --tpotparam=$tpotparam)
echo "$all_output"
output=$(echo "$all_output" | tail -n 1)

if [[ "$output" == *"RUN INCOMPLETE"* ]]; then
    echo "RUNNING NEXT GENERATION..."
    outputid=${output##* }
    bash $0 -i "$outputid"
elif [[ "$output" == "RUN COMPLETE" ]]; then
    echo "ENDING RECURSION"
else
    echo "UNEXPECTED OUTPUT:" "$output"
fi

