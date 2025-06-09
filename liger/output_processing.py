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


from typing import Any, Sequence, overload
from pathlib import Path
import json


def _json_load(filepath: Path) -> Any:
    with open(filepath, "r") as file:
        return json.load(file)


def mass_json_load(
    paths: Path | str | Sequence[Path | str],
    pattern: str = "pipeline_data.json",
) -> Any:
    if isinstance(paths, (Path, str)):
        paths = paths,
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"Warning: {path} does not exist")
            continue
        if path.is_file() and path.match(pattern) and path.suffix == ".json":
            yield _json_load(path)
        for fpath in path.rglob(pattern):
            if not fpath.suffix == ".json":
                continue
            yield _json_load(fpath)


_sentinel = object()

@overload
def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str) -> list[Any]: ...
@overload
def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str, default: Any) -> list[Any]: ...

def mass_dict_get(dicts: Sequence[dict[str, Any]], key: str, default: Any = _sentinel) -> list[Any]:
    if default is _sentinel:
        return [dictionary.get(key) for dictionary in dicts]
    return [dictionary.get(key, default) for dictionary in dicts]


def is_run_finished(output: dict[str, Any]) -> bool:
    kfold_scores = output["pipeline_attributes"]["kfold_scores"]
    if isinstance(kfold_scores, dict):
        return kfold_scores != {}
    raise TypeError("\"kfold_scores\" in output is not of type dict")


def list_unfinished_runs(
    paths: Path | str | Sequence[Path | str],
    read_all_json_files: bool = False,
    filenames_to_read: str | set[str] = "pipeline_data.json",
) -> list[str]:
    return sorted([
        output["pipeline_parameters"]["id"]
        for output in mass_json_load(paths, read_all_json_files, filenames_to_read)
        if not is_run_finished(output)
    ])


