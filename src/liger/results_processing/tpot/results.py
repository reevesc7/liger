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
import warnings
from functools import cached_property
import pandas as pd
from liger.results_processing.json_cache import JSONCache


class TPOTResults:
    def __init__(self, cache: JSONCache) -> None:
        self._cache = cache
        self.runs = pd.DataFrame([run_data for _, run_data in self._cache]).set_index(
            "run_id"
        )
        self.individuals = pd.concat(
            self.runs.pop("evaluated_individuals")
            .map(self._format_individuals)
            .tolist(),
            keys=self.runs.index,
        )
        nodes = self.individuals.pop("Instance").map(self._decomp_pipeline).explode()
        self.nodes = pd.DataFrame(
            nodes.tolist(),
            index=nodes.index,
        ).set_index("node_id", append=True)
        self.edges = self.nodes.pop("successors").explode().rename("source_id")

    @staticmethod
    def _format_individuals(individuals: dict[str, Any]) -> pd.DataFrame:
        formatted = pd.DataFrame(individuals)
        formatted.index.name = "individual_id"
        formatted.index = formatted.index.astype(int)
        formatted["Generation"] = formatted["Generation"].astype(int)
        formatted = formatted.reset_index().set_index(["Generation", "individual_id"])
        return formatted

    @staticmethod
    def _decomp_pipeline(individual: dict[str, Any]) -> list[dict[str, Any]]:
        # TODO: add handling of pipelines other than GraphPipeline
        if individual["method"] != "tpot.graphsklearn.GraphPipeline":
            warnings.warn(
                f"{individual['method']} is not a GraphPipeline; skipping",
                UserWarning,
            )
            return []
        graph = individual["params"]["graph"]
        if not isinstance(graph, dict):
            warnings.warn(
                f"Invalid node graph:\n{individual['params']['graph']}",
                UserWarning,
            )
            return []
        return [
            {
                "node_id": node_id,
                "method": node_data["instance"]["method"],
                "params": node_data["instance"]["params"],
                "successors": node_data["successors"],
            }
            for node_id, node_data in graph.items()
        ]

    @staticmethod
    def _run_is_finished(run_data: Any) -> bool:
        kfold_scores = run_data["manager_attributes"]["kfold_scores"]
        if isinstance(kfold_scores, dict):
            return kfold_scores != {}
        raise TypeError('"kfold_scores" in output is not of type dict')

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
