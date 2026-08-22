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


from pathlib import Path
from argparse import ArgumentParser, Namespace
from enum import StrEnum


class _Consumer(StrEnum):
    TPOT = "tpot"


class _RunCommand(StrEnum):
    INIT = "init"
    RUN = "run"


class _RunMode(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"


def _cmd_tpot_init(args: Namespace) -> None:
    from liger.run import tpot
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path!r} is not a file")
    if args.sprofile is not None:
        slurm_profile_path = Path(args.sprofile)
        if not slurm_profile_path.is_file():
            raise FileNotFoundError(f"{slurm_profile_path!r} is not a file")
    else:
        slurm_profile_path = None
    n = 1 if args.n is None or args.n < 1 else args.n
    checkpoint_dirs = [
        tpot.init_tpot_dir(config_path, slurm_profile_path, args.output)
        for _ in range(n)
    ]
    if args.run is None:
        return
    if args.run == _RunMode.LOCAL:
        for dir in checkpoint_dirs:
            tpot.run_local_segment(dir)
        return
    if args.run == _RunMode.SLURM:
        for dir in checkpoint_dirs:
            tpot.run_slurm_segment(dir, args.recurse)
        return
    raise ValueError(f"{args.run!r} is not a valid run mode")


def _cmd_tpot_local_run(args: Namespace) -> None:
    from liger.run import tpot
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"{checkpoint_dir!r} is not a directory")
    tpot.run_local_segment(checkpoint_dir)


def _cmd_tpot_slurm_run(args: Namespace) -> None:
    from liger.run import tpot
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"{checkpoint_dir!r} is not a directory")
    tpot.run_slurm_segment(checkpoint_dir, args.recurse)


def _init_tpot_init_parser(tpot_init_parser: ArgumentParser) -> None:
    tpot_init_parser.add_argument("config", type=Path, help="Path of the config file.")
    tpot_init_parser.add_argument("--sprofile", type=Path, help=(
        "Path of a shell script for configuring Slurm jobs. "
            "Script must only activate a Python environment with liger installed. "
            "SBATCH options can also be specified as comments within the script "
            "(e.g., '#SBATCH --mem-per-cpu=32G')."
    ))
    tpot_init_parser.add_argument("-n", type=int, help=(
        "Number of directories to initialize."
    ))
    tpot_init_parser.add_argument("-o", "--output", type=Path, help=(
        "Output/checkpoint directory, for all output files and checkpoints. "
            "Overrides output directory specification in config file."
    ))
    tpot_init_parser.add_argument(
        "--run",
        choices=[member.value for member in _RunMode],
        help="(run mode) Immediately run a segment from the initialized checkpoint. "
            "Specify a run mode and any arguments associated with the mode.",
    )
    tpot_init_parser.add_argument("--recurse", action="store_true", help=(
        f"Only used if '--run {_RunMode.SLURM}'. See 'liger tpot run slurm --recurse'."
    ))
    tpot_init_parser.set_defaults(func=_cmd_tpot_init)


def _init_tpot_run_parser(tpot_run_parser: ArgumentParser) -> None:
    run_common = ArgumentParser(add_help=False)
    run_common.add_argument("checkpoint", type=Path, help=(
        "Path of the checkpoint directory."
    ))
    run_subparsers = tpot_run_parser.add_subparsers(dest="mode", required=True)
    local_run_parser = run_subparsers.add_parser(_RunMode.LOCAL, parents=[run_common])
    local_run_parser.set_defaults(func=_cmd_tpot_local_run)
    slurm_run_parser = run_subparsers.add_parser(_RunMode.SLURM, parents=[run_common])
    slurm_run_parser.add_argument("--recurse", action="store_true", help=(
        "Automatically and recursively submit "
            "follow-up segments as Slurm jobs until the run is complete."
    ))
    slurm_run_parser.set_defaults(func=_cmd_tpot_slurm_run)


def _init_tpot_parser(tpot_parser: ArgumentParser) -> None:
    subparsers = tpot_parser.add_subparsers(dest="command", required=True)
    _init_tpot_init_parser(subparsers.add_parser(
        _RunCommand.INIT,
        description="Initialize one or more TPOT run output/checkpoint directories "
            "from a config.",
    ))
    _init_tpot_run_parser(subparsers.add_parser(
        _RunCommand.RUN,
        description="Run a TPOT segment or a chain of segments from a checkpoint.",
        add_help=False,
    ))


def _make_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Start a liger TPOT run")
    subparsers = parser.add_subparsers(dest="consumer", required=True)
    _init_tpot_parser(subparsers.add_parser(_Consumer.TPOT, description=(
        "TPOT run commands"
    )))
    return parser


def main():
    parser = _make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
