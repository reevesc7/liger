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
from ..output_processing import JSONCache


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


class DemographicsAnalyzer(TPOTOuputAnalyzer):
    @staticmethod
    def _decomp_pipeline(
        individual: pd.Series,
    ) -> list[dict[str, str]]:
        # TODO: add handling of pipelines other than GraphPipeline
        if individual["Instance"]["method"] != "tpot.graphsklearn.GraphPipeline":
            print(f"Individual {individual.name} is not a GraphPipeline or was stored improperly")
            return []
        graph = individual["Instance"]["params"]["graph"]
        if not isinstance(graph, dict):
            print(f"Individual{individual.name} does not have a valid node graph")
            return []
        connections: list[dict[str, str]] = []
        for node_id, node_data in graph.items():
            if len(node_data["successors"]) == 0:
                connections.append({
                    "node_id": node_id,
                    "node_method": node_data["instance"]["method"],
                    "upstream_id": "raw_features",
                })
                continue
            for upstream_id in node_data["successors"]:
                connections.append({
                    "node_id": node_id,
                    "node_method": node_data["instance"]["method"],
                    "upstream_id": upstream_id,
                })
        return connections

    @cached_property
    def connections(self) -> pd.DataFrame:
        individuals = self.individuals.copy()
        individuals["Instance"] = individuals.apply(self._decomp_pipeline, axis=1)
        connections = individuals.explode("Instance").reset_index()
        return connections.join(pd.DataFrame(connections.pop("Instance").tolist()))

    @cached_property
    def _indiv_connections(self) -> pd.api.typing.DataFrameGroupBy:
        return self.connections.groupby(["run_id", "individual_id"])

    @cached_property
    def n_nodes(self) -> pd.Series:
        return pd.Series(self._indiv_connections["node_id"].nunique(), name="n_nodes")

    @cached_property
    def n_branches(self) -> pd.Series:
        return pd.Series(self._indiv_connections.size() - self.n_nodes, name="n_branches")

    @cached_property
    def n_nodes_of_method(self) -> pd.Series:
        nodes = self.connections.drop_duplicates(subset=[
            "run_id",
            "individual_id",
            "node_id",
        ])
        return pd.Series(nodes.groupby([
            "run_id",
            "individual_id",
            "node_method",
        ])["node_method"].count(), name="n_nodes_of_method")

    @cached_property
    def n_leaves_of_method(self) -> pd.Series:
        leaves = self.connections.loc[self.connections["upstream_id"] == "raw_features"]
        return pd.Series(leaves.groupby([
            "run_id",
            "individual_id",
            "node_method",
        ])["node_method"].count(), name="n_leaves_of_method")

    @cached_property
    def root_methods(self) -> pd.Series:
        return pd.Series(self.individuals["Instance"].apply(
            lambda entry: next(iter(entry["params"]["graph"].values()))["instance"]["method"]
        ), name="root_node")

