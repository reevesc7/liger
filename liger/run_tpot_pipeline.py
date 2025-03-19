from typing import Any
from argparse import ArgumentParser, Namespace
from liger.pipelines.tpot import TPOTPipeline


def main():
    args = get_args()

    checkpoint_file = None
    id = parse_arg_or_none(args.id),
    if id is not None:
        id = str(id)
        checkpoint_file = TPOTPipeline.find_checkpoint(id)
    if checkpoint_file is not None:
        pipeline = TPOTPipeline.from_checkpoint(checkpoint_file)
    else:
        config_file = parse_arg_or_none(args.config)
        if config_file is None:
            raise ValueError("Config file must be specified for new runs!")
        pipeline = TPOTPipeline(
            config_file=config_file,
            data_file=parse_arg_or_none(args.data),
            tpot_random_state=parse_arg_or_none(args.tpotrs),
            slurm_id=args.slurmid,
            id=id,
        )
    pipeline.run_1_gen()


def parse_arg_or_none(arg: Any | None) -> Any | None:
    if arg is None or str(arg).lower() in ["", "none", "(none,)"]:
        return None
    return arg


def get_args() -> Namespace:
    parser = ArgumentParser(description="Script that runs tpot fitting and evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="config file path"
    )       #e.g., "Configs/nolong_reg_1.json"
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default=None,
        help="training data file path"
    )       #e.g., "Data/human_size_rating_1_1.csv"
    parser.add_argument(
        "--tpotrs",
        required=False,
        default=None,
        help="tpot random state (int)"
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

