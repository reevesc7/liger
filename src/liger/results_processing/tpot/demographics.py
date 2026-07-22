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


from functools import cached_property
import pandas as pd
from liger.results_processing.tpot.results import TPOTResults


class TPOTDemographics(TPOTResults):
    @cached_property
    def n_nodes(self) -> pd.Series:
        return pd.Series(
            self.nodes.groupby(["run_id", "Generation", "individual_id"]).size(),
            name="n_nodes",
        )

    @cached_property
    def leaves(self) -> pd.DataFrame:
        return pd.DataFrame(self.nodes.loc[self.edges.loc[self.edges.isna()].index])

    @cached_property
    def n_leaves(self) -> pd.Series:
        return pd.Series(self.leaves.groupby([
            "run_id",
            "Generation",
            "individual_id",
        ]).size())

    @cached_property
    def n_branches(self) -> pd.Series:
        return pd.Series(
            self.edges.groupby([
                "run_id",
                "Generation",
                "individual_id",
            ]).size() - self.n_nodes,
            name="n_branches",
        )

    @cached_property
    def n_nodes_of_method(self) -> pd.Series:
        return pd.Series(self.nodes.groupby([
            "run_id",
            "Generation",
            "individual_id",
            "method",
        ]).size(), name="n_nodes_of_method")

    @cached_property
    def n_leaves_of_method(self) -> pd.Series:
        return pd.Series(self.leaves.groupby([
            "run_id",
            "Generation",
            "individual_id",
            "method",
        ]).size(), name="n_leaves_of_method")

    @cached_property
    def root_methods(self) -> pd.Series:
        return pd.Series(
            self.nodes["method"],
        ).groupby(["run_id", "Generation", "individual_id"]).first().rename("root_node")
