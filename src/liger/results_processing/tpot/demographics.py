import warnings
from functools import cached_property
import pandas as pd
from liger.results_processing.tpot.base import TPOTOuputAnalyzer


class DemographicsAnalyzer(TPOTOuputAnalyzer):
    @staticmethod
    def _decomp_pipeline(
        individual: pd.Series,
    ) -> list[dict[str, str]]:
        # TODO: add handling of pipelines other than GraphPipeline
        if individual["Instance"]["method"] != "tpot.graphsklearn.GraphPipeline":
            warnings.warn(
                f"Individual {individual.name} is not a GraphPipeline; skipping",
                UserWarning,
            )
            return []
        graph = individual["Instance"]["params"]["graph"]
        if not isinstance(graph, dict):
            warnings.warn(
                f"Individual {individual.name} does not have a valid node graph; skipping",
                UserWarning,
            )
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

