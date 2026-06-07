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
from functools import cached_property
import pandas as pd
from liger.results_processing.json_cache import JSONCache


class TPOTOuputAnalyzer:
    def __init__(self, cache: JSONCache) -> None:
        self._cache = cache

    @staticmethod
    def _run_is_finished(run_data: Any) -> bool:
        kfold_scores = run_data["manager_attributes"]["kfold_scores"]
        if isinstance(kfold_scores, dict):
            return kfold_scores != {}
        raise TypeError("\"kfold_scores\" in output is not of type dict")

    @cached_property
    def finished_runs(self):
        return {
            filepath: data
            for filepath, data in self._cache
            if self._run_is_finished(data)
        }

    @cached_property
    def unfinished_runs(self):
        return {
            filepath: data
            for filepath, data in self._cache
            if filepath not in self.finished_runs
        }

    @cached_property
    def individuals(self) -> pd.DataFrame:
        runs_individuals: list[pd.DataFrame] = []
        for run_id, run_data in self._cache:
            print(f"Loading {run_id}")
            individuals = pd.DataFrame(run_data["tpot_attributes"]["evaluated_individuals"]).drop([
                "Parents",
                "Variation_Function",
                "Submitted Timestamp",
                "Completed Timestamp",
                "Eval Error",
            ], axis=1)
            individuals["run_id"] = run_id
            individuals.index.name = "individual_id"
            individuals = individuals.reset_index()
            runs_individuals.append(individuals)
        return pd.concat(runs_individuals, axis=0).set_index(["run_id", "individual_id"])

