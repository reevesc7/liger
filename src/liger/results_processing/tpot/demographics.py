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
    RUN_GEN_IND = ["run_id", "Generation", "individual_id"]
    RUN_GEN_IND_METHOD = RUN_GEN_IND + ["method"]

    @cached_property
    def n_nodes(self) -> pd.Series:
        n_nodes = self.nodes.groupby(self.RUN_GEN_IND).size()
        n_nodes.name = "n_nodes"
        return n_nodes

    @cached_property
    def leaves(self) -> pd.DataFrame:
        return self.nodes.loc[self.edges.loc[self.edges.isna()].index]

    @cached_property
    def n_leaves(self) -> pd.Series:
        n_leaves = self.leaves.groupby(self.RUN_GEN_IND).size()
        n_leaves.name = "n_leaves"
        return n_leaves

    @cached_property
    def n_branches(self) -> pd.Series:
        n_branches = self.edges.groupby(self.RUN_GEN_IND).size() - self.n_nodes
        n_branches.name = "n_branches"
        return n_branches

    @cached_property
    def n_nodes_of_method(self) -> pd.Series:
        n_nodes_of_method = self.nodes.groupby(self.RUN_GEN_IND_METHOD).size()
        n_nodes_of_method.name = "n_nodes_of_method"
        return n_nodes_of_method

    @cached_property
    def n_leaves_of_method(self) -> pd.Series:
        n_leaves_of_method = self.leaves.groupby(self.RUN_GEN_IND_METHOD).size()
        n_leaves_of_method.name = "n_leaves_of_method"
        return n_leaves_of_method

    @cached_property
    def root_methods(self) -> pd.Series:
        root_methods = self.nodes["method"].groupby(self.RUN_GEN_IND).first()
        root_methods.name = "root_node"
        return root_methods
