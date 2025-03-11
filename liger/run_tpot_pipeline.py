from argparse import ArgumentParser, Namespace
from liger.pipelines.tpot import TPOTPipeline


def main():
    args = get_args()

    data_name = TPOTPipeline.get_filename(args.data)
    if str(args.id).lower() in {"", "none"}:
        id = None
    else:
        id = args.id
    pickle_file = None
    if id is not None:
        pickle_file = TPOTPipeline.find_pickle(data_name, args.id)
    if pickle_file is not None:
        pipeline = TPOTPipeline.from_pickle(pickle_file)
    else:
        if args.nevalrs is None:
            eval_random_states = None
        else:
            eval_random_states = [i+args.sevalrs for i in range(args.nevalrs)]
        pipeline = TPOTPipeline(
            config_file=args.config,
            data_file=args.data,
            tpot_random_state=args.tpotrs,
            slurm_id=args.slurmid,
            id=id,
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
        "--tpotrs",
        type=int,
        required=False,
        default=None,
        help="tpot random state"
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

