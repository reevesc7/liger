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


from typing import Any, Iterable, Iterator
import warnings
from pathlib import Path
import json
import dill
import pandas as pd
from tpot import Population


def _is_subset(sub: Any, super: Any) -> bool:
    if sub == super:
        return True
    if isinstance(sub, dict) and isinstance(super, dict):
        for sub_key, sub_value in sub.items():
            if sub_key not in super:
                return False
            if not _is_subset(sub_value, super[sub_key]):
                return False
        return True
    if isinstance(sub, list) and isinstance(super, list):
        if len(sub) > len(super):
            return False
        return all(_is_subset(sub_e, super_e) for sub_e, super_e in zip(sub, super))
    return False


def _passes_filters(run_data: Any, filters: Iterable[dict[str, Any]]) -> bool:
    """Check whether a run passes the run filters.
    #
    A run must pass at least one filter (OR).
    A filter is passed if it is a subset of the run's data (AND).
    """
    if not any(_is_subset(filter, run_data) for filter in filters):
        return False
    return True


def load_data(
    path: Path | str,
    filters: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    warnings.warn(
        "Use liger.results_processing.json_cache.JSONCache instead",
        DeprecationWarning,
    )
    path = Path(path)
    manager_path = path / "manager_data.json"
    population_path = path / "population.pkl"
    temp_population_path = path / "temp-population.pkl"
    if not manager_path.is_file() or temp_population_path.is_file():
        return
    with open(manager_path, "r") as file:
        manager_data: dict[str, Any] = json.load(file)
    if not manager_data:
        warnings.warn(
            f"{manager_path} is an empty file; skipping",
            UserWarning,
        )
        return
    if filters is not None and not _passes_filters(manager_data, filters):
        return
    eval_indivs = pd.DataFrame(manager_data["tpot_attributes"]["evaluated_individuals"])
    if population_path.is_file():
        with open(population_path, "rb") as file:
            population: Population = dill.load(file)
        eval_indivs["Individual"] = (
            population.evaluated_individuals["Individual"].to_numpy()
        )
    manager_data["tpot_attributes"]["evaluated_individuals"] = eval_indivs
    return manager_data


def mass_load_data(
    path: Path | str,
    pattern: str,
    filters: Iterable[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    warnings.warn(
        "Use liger.results_processing.json_cache.JSONCache instead",
        DeprecationWarning,
    )
    path = Path(path)
    subdirs = sorted([dpath for dpath in path.rglob(pattern) if dpath.is_dir()])
    manager_data = load_data(path, filters)
    if manager_data is not None:
        yield manager_data
    for subdir in subdirs:
        manager_data = load_data(subdir, filters)
        if manager_data is not None:
            yield manager_data
