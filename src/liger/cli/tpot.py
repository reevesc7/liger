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
import shutil
from pathlib import Path
import subprocess
from argparse import ArgumentParser, Namespace
from enum import Enum, auto
import json
from liger.cli.core import get_script_path
from liger.training.tpot import TPOTManager


SCRIPT_DIR = Path("tpot")
SLURM_SCRIPT = SCRIPT_DIR / "tpot_segment.sb"
SLURM_INIT = Path("slurm_init.sh")
SLURM_FLAGS = Path("slurm_flags.txt")


class _ExecMode(Enum):
    LOCAL = auto()
    SLURM = auto()


class _JSONObject(dict):
    def __init__(self, string: str) -> None:
        obj = json.loads(string)
        if not isinstance(obj, dict):
            raise ValueError(
                f"JSON object required, received {type(obj).__name__}: {string!r}"
            )
        super().__init__(obj)


def _parse_slurm_options(filepath: Path) -> list[str]:
    if not filepath.is_file():
        return []
    lines = [line for line in filepath.read_text().split("\n") if "SBATCH" in line]
    return [line.rsplit("SBATCH ")[-1] for line in lines]


def _queue_segment(
    checkpoint_dir: Path,
    exec_mode: _ExecMode,
) -> None:
    checkpoint_dir = checkpoint_dir.resolve()
    if exec_mode == _ExecMode.LOCAL:
        subprocess.Popen(["lgtpot", "checkpoint", checkpoint_dir, "--recurse"])
        return
    slurm_options = _parse_slurm_options(
        checkpoint_dir / SLURM_FLAGS
    ) + [f"--output={checkpoint_dir}/slurm-%j.out"]
    with get_script_path(SLURM_SCRIPT) as script:
        subprocess.run(["sbatch"] + slurm_options + [script, checkpoint_dir])


def _config_command(args: Namespace) -> None:
    if args.sinit is not None:
        exec_mode = _ExecMode.SLURM
    else:
        exec_mode = _ExecMode.LOCAL
    if args.kwargs is None:
        kwargs: dict[str, Any] = {}
    else:
        kwargs = args.kwargs
    if args.dir is not None:
        kwargs["output_dir"] = args.dir
    if args.rstate is not None:
        kwargs["random_state"] = args.rstate
    tpot = TPOTManager.from_config(args.configpath, **kwargs)
    if args.sinit is not None:
        shutil.copy(args.sinit, tpot.output_dir / SLURM_INIT)
        if args.slurmcfg is not None:
            shutil.copy(args.sopts, tpot.output_dir / SLURM_FLAGS)
    if args.recurse:
        _queue_segment(tpot.output_dir, exec_mode)


def _init_config_parser(parser: ArgumentParser) -> None:
    parser.set_defaults(func=_config_command)
    parser.add_argument(
        "configpath",
        type=Path,
        help="(Path) Path to the config file",
    )
    parser.add_argument(
        "-r",
        "--rstate",
        type=int,
        help="(int) Random state for the run; overrides config file random state"
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=Path,
        help=(
            "(Path) Run directory, for all output files and checkpoints; "
            "overrides config file output directory"
        ),
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help=(
            "Automatically and recursively submit segments "
            "after setting up the output directory"
        ),
    )
    parser.add_argument(
        "--sinit",
        type=Path,
        help=(
            "(Path) Path to a Python environment setup shell script for Slurm jobs; "
            "if provided, run segments *can* be submitted as Slurm jobs; "
            "if `--recurse` is present, "
            "automatic segments *will* be submitted as Slurm jobs"
        ),
    )
    parser.add_argument(
        "--sopts",
        type=Path,
        help=(
            "(Path) Path to a text file with SBATCH options; "
            "provides `sbatch` options for segments as Slurm jobs; "
            "ignored if `--sinit` is not set"
        ),
    )
    parser.add_argument(
        "-k",
        "--kwargs",
        type=_JSONObject,
        help="Keyword arguments to pass to TPOTManager.from_config()",
    )


def _checkpoint_command(args: Namespace) -> None:
    tpot = TPOTManager.from_checkpoint(args.checkpointdir)
    tpot.run_segment()
    if args.slurm:
        exec_mode = _ExecMode.SLURM
    else:
        exec_mode = _ExecMode.LOCAL
    if args.recurse and not tpot.is_complete():
        _queue_segment(tpot.output_dir, exec_mode)


def _init_checkpoint_parser(parser: ArgumentParser) -> None:
    parser.set_defaults(func=_checkpoint_command)
    parser.add_argument(
        "checkpointdir",
        type=Path,
        help="(Path) Path to the directory containing checkpoint files",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help=(
            "Automatically and recursively submit follow-up segments after this one"
        ),
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Submit any follow-up segments as Slurm jobs",
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Start a liger TPOT run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _init_config_parser(subparsers.add_parser(
        "config",
        description="Set up a TPOT output directory from a config file",
    ))
    _init_checkpoint_parser(subparsers.add_parser(
        "checkpoint",
        description="Run a TPOT segment from a checkpoint directory",
    ))
    return parser.parse_args()




def main():
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
