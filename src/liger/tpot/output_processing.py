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
from pathlib import Path
from ..output_processing import mass_json_load


def is_run_finished(output: dict[str, Any]) -> bool:
    kfold_scores = output["manager_attributes"]["kfold_scores"]
    if isinstance(kfold_scores, dict):
        return kfold_scores != {}
    raise TypeError("\"kfold_scores\" in output is not of type dict")


def list_unfinished_runs(
    paths: Path | str | Iterable[Path | str],
    filename_pattern: str = "manager_data.json",
) -> list[str]:
    return sorted([
        output["manager_parameters"]["id"]
        for output in mass_json_load(paths, filename_pattern)
        if not is_run_finished(output)
    ])


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


def _passes_run_filters(run_data: Any, filters: Iterable[Any]) -> bool:
    """Check whether a run passes the run filters.
    #
    A run must pass at least one filter (OR).
    A filter is passed if it is a subset of the run's data (AND).
    """
    if not any(_is_subset(filter, run_data) for filter in filters):
        return False
    return True


def filtered_runs(
        paths: Path | str | Iterable[Path | str],
        id_filter: str,
        data_filters: Iterable[Any],
) -> Iterator[Any]:
    """Yield runs which pass the run filters.
    #
    To be returned, a run must pass at least one filter (OR).
    A filter is passed if it is a subset of the run's data (AND).
    #
    Parameters
    ----------
    `paths` : `Path | str | Iterable[Path | str]`
        The directory/ies to search recursively for `manager_data.json` files.
    `id_filter` : `str`
        The pattern to match run IDs to.
    `data_filters` : `Iterable[Any]`
        The filters to match run data to. A run must pass at least one of the filters,
        and a filter matches when it is a subset of the run's data.
    #
    Returns
    -------
    `filtered_runs` : `Iterator[Any]`
        An iterator of run data which passed the ID and data filters.
    """
    for run_data in mass_json_load(paths, id_filter):
        if not _passes_run_filters(run_data, data_filters):
            continue
        yield run_data

