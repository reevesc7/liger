# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Any
from argparse import ArgumentParser, Namespace
import json
from liger.pipelines.tpot import TPOTPipeline


def main():
    args = get_args()

    checkpoint = None
    if args.id is not None:
        checkpoint = TPOTPipeline.find_checkpoint(args.id)
    if checkpoint is not None:
        pipeline = TPOTPipeline.from_checkpoint(checkpoint, args.slurmid)
    else:
        if args.config is None:
            raise ValueError("Config file must be specified for new runs!")
        pipeline = TPOTPipeline(
            config_file=args.config,
            data_file=args.data,
            tpot_random_state=args.tpotrs,
            pipeline_parameters=args.pipeparam,
            tpot_parameters=args.tpotparam,
            slurm_id=args.slurmid,
            id=args.id,
        )
    pipeline.run_1_gen()


def str_or_none(arg: Any | None) -> str | None:
    if arg is None or str(arg).lower() in ["", "none"]:
        return None
    return str(arg)


def int_or_none(arg: Any | None) -> int | None:
    if arg is None or str(arg).lower() in ["", "none"]:
        return None
    return int(arg)


def dict_or_none(arg: Any | None) -> dict | None:
    if arg is None or str(arg).lower() in ["", "none"]:
        return None
    return json.loads(arg)


def get_args() -> Namespace:
    parser = ArgumentParser(description="Script that runs tpot fitting and evaluation")
    parser.add_argument(
        "-c",
        "--config",
        type=str_or_none,
        required=False,
        help="config file path",
    )       #e.g., "Configs/nolong_reg_1.json"
    parser.add_argument(
        "-d",
        "--data",
        type=str_or_none,
        required=False,
        default=None,
        help="training data file path",
    )       #e.g., "Data/human_size_rating_1_1.csv"
    parser.add_argument(
        "-r",
        "--tpotrs",
        type=int_or_none,
        required=False,
        default=None,
        help="tpot random state (int)",
    )
    parser.add_argument(
        "--slurmid",
        type=int,
        required=False,
        default=None,
        help="$SLURM_JOB_ID, if applicable",
    )
    parser.add_argument(
        "-i",
        "--id",
        type=str_or_none,
        required=False,
        default=None,
        help="ID of the job - used for all output files",
    )
    parser.add_argument(
        "-p",
        "--pipeparam",
        type=dict_or_none,
        required=False,
        default='{}',
        help="JSON formatted dict of any parameters to pass to pipeline",
    )
    parser.add_argument(
        "-t",
        "--tpotparam",
        type=dict_or_none,
        required=False,
        default='{}',
        help="JSON formatted dict of any parameters to pass to TPOTEstimator",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

