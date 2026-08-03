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


import shutil
from pathlib import Path
import subprocess
from argparse import ArgumentParser, Namespace
from enum import IntEnum, auto
import json
from datetime import datetime, timezone
from liger.typing import LgConfig
import liger.config as cfg
from liger.tpot import TPOTManager
from ._script_path import get_script_path


DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
SCRIPT_DIR = Path("")
SLURM_SCRIPT = SCRIPT_DIR / "tpot_segment.sb"
SLURM_INIT_NAME = Path("slurm_init.sh")
SLURM_OPTS_NAME = Path("slurm_opts.txt")
CONFIG_NAME = Path("config.json")
PARAMS_NAME = Path("params.json")


class _ExecMode(IntEnum):
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
        subprocess.Popen(["lgtpot", "segment", checkpoint_dir, "--recurse"])
        return
    slurm_options = _parse_slurm_options(
        checkpoint_dir / SLURM_OPTS_NAME
    ) + [f"--output={checkpoint_dir}/slurm-%j.out"]
    with get_script_path(SLURM_SCRIPT) as script:
        subprocess.run(["sbatch"] + slurm_options + [script, checkpoint_dir])


def _init_command(args: Namespace) -> None:
    if args.sinit is not None:
        exec_mode = _ExecMode.SLURM
    else:
        exec_mode = _ExecMode.LOCAL
    config: dict[str, LgConfig] = json.load(args.configpath.open())
    now = datetime.now(timezone.utc).strftime(DATETIME_FMT)
    if args.dir is not None:
        config["output_dir"] = args.dir
    elif "output_dir" in config:
        config["output_dir"] = str(Path(str(config["output_dir"])) / now)
    else:
        config["output_dir"] = now
    if args.rstate is not None:
        config["random_state"] = args.rstate
    tpot = cfg.parse_config(config, TPOTManager)
    tpot.output_dir_.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.configpath, tpot.output_dir_ / CONFIG_NAME)
    json.dump(cfg.to_config(tpot), (tpot.output_dir_ / PARAMS_NAME).open("w"), indent=4)
    if args.sinit is not None:
        shutil.copy(args.sinit, tpot.output_dir_ / SLURM_INIT_NAME)
        if args.slurmcfg is not None:
            shutil.copy(args.sopts, tpot.output_dir_ / SLURM_OPTS_NAME)
    if args.recurse:
        _queue_segment(tpot.output_dir_, exec_mode)


def _make_init_parser(parser: ArgumentParser) -> None:
    parser.set_defaults(func=_init_command)
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
        help="(Path) Run directory, for all output files and checkpoints; "
            "overrides config file output directory",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help=(
            "Automatically and recursively submit segments "
                "after setting up the output/checkpoint directory"
        ),
    )
    parser.add_argument(
        "--sinit",
        type=Path,
        help="(Path) Path to a Python environment setup shell script for Slurm jobs; "
            "if provided, run segments *can* be submitted as Slurm jobs; "
            "if `--recurse` is present, "
            "automatic segments *will* be submitted as Slurm jobs",
    )
    parser.add_argument(
        "--sopts",
        type=Path,
        help="(Path) Path to a text file with SBATCH options; "
            "provides `sbatch` options for segments as Slurm jobs; "
            "ignored if `--sinit` is not set",
    )
    parser.add_argument(
        "-k",
        "--kwargs",
        type=_JSONObject,
        help="Keyword arguments to pass to the 'TPOTManager' constructor specified "
            "in the config file top-level '<instance>' field",
    )


def _segment_command(args: Namespace) -> None:
    checkpoint_dir = Path(args.checkpointdir).resolve()
    config: dict[str, LgConfig] = json.load((checkpoint_dir / PARAMS_NAME).open())
    if Path(str(config["output_dir"])).resolve() != checkpoint_dir:
        config["output_dir"] = str(checkpoint_dir)
        json.dump(config, checkpoint_dir.open("w"), indent=4)
    tpot = cfg.parse_config(config, TPOTManager)
    tpot.run_segment()
    if args.slurm:
        exec_mode = _ExecMode.SLURM
    else:
        exec_mode = _ExecMode.LOCAL
    if args.recurse and not tpot.is_complete_:
        _queue_segment(tpot.output_dir_, exec_mode)


def _make_segment_parser(parser: ArgumentParser) -> None:
    parser.set_defaults(func=_segment_command)
    parser.add_argument(
        "checkpointdir",
        type=Path,
        help="(Path) Path to the output/checkpoin directory",
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


def _parse_args() -> Namespace:
    parser = ArgumentParser(description="Start a liger TPOT run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _make_init_parser(subparsers.add_parser(
        "init",
        description="Initialize a TPOT output/checkpoint directory from a config file",
    ))
    _make_segment_parser(subparsers.add_parser(
        "segment",
        description="Run a TPOT segment from a checkpoint directory",
    ))
    return parser.parse_args()


def main():
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
