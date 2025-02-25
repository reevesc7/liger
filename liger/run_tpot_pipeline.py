from argparse import ArgumentParser, Namespace
from liger.pipelines.tpot import TPOTPipeline


def main():
    args = get_args()

    data_name = TPOTPipeline.get_data_name(args.data)
    id = TPOTPipeline.get_id(args.ngens, args.popsize, args.tpotrs, args.regression, args.notrees, args.noxg)
    pickle_file = TPOTPipeline.find_pickle(data_name, id)
    if pickle_file:
        pipeline = TPOTPipeline.from_pickle(pickle_file)
    else:
        if args.sevalrs is None or args.nevalrs is None:
            eval_random_states = None
        else:
            eval_random_states = [i+args.sevalrs for i in range(args.nevalrs)]
        pipeline = TPOTPipeline(
            config_file=args.config,
            data_file=args.data,
            target_gens=args.ngens,
            population_size=args.popsize,
            tpot_random_state=args.tpotrs,
            eval_random_states=eval_random_states,
            max_time_mins=args.maxtime,
            n_splits=args.nsplits,
            slurm_id=args.slurmid,
            id=args.id,
        )
    pipeline.run_1_gen()


def get_args() -> Namespace:
    parser = ArgumentParser(description="Script that runs tpot fitting and evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file path"
    )       #e.g., "Configs/nolong_reg_1.json"
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="training data file path"
    )       #e.g., "Data/human_size_rating_1_1.csv"
    parser.add_argument(
        "--ngens",
        type=int,
        required=False,
        default=None,
        help="number of tpot fitting generations"
    )
    parser.add_argument(
        "--popsize",
        type=int,
        required=False,
        default=None,
        help="tpot fitting population size"
    )
    parser.add_argument(
        "--tpotrs",
        type=int,
        required=False,
        default=None,
        help="tpot random state"
    )
    parser.add_argument(
        "--sevalrs",
        type=int,
        required=False,
        default=None,
        help="eval random state start"
    )
    parser.add_argument(
        "--nevalrs",
        type=int,
        required=False,
        default=None,
        help="number of eval random states to test"
    )
    parser.add_argument(
        "--maxtime",
        type=int,
        required=False,
        default=None,
        help="minutes at which tpot should terminate fitting early"
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        required=False,
        default=None,
        help="number of splits for K-fold validation (keep blank for leave one out)"
    )
    parser.add_argument(
        "--slurmid",
        type=int,
        required=False,
        default=None,
        help="$SLURM_JOB_ID, if applicable"
    )
    parser.add_argument(
        "--id",
        type=str,
        required=False,
        default=None,
        help="ID of the job - used for all output files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

