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


from sklearn.base import BaseEstimator
from sklearn.metrics._scorer import _Scorer
# from sklearn.pipeline import Pipeline
# from networkx.classes import DiGraph
# from tpot.graphsklearn import GraphPipeline
from liger.typing import LgConfig
import liger.config as cfg
from ._manager import TPOTManager
from ._params import (
    Objective,
    InverseObjectives,
    DatasetParams,
    EvolutionParams,
    EvalParams,
    EndParams,
    RuntimeParams,
)


def init_digraph(
    graph_attrs: dict,
    node_attrs: list,
    edge_attrs: list,
) -> DiGraph:
    digraph = DiGraph(**graph_attrs)
    digraph.add_nodes_from(node_attrs)
    digraph.add_edges_from(edge_attrs)
    return digraph


def register() -> None:
    @cfg.to_config.register(
        BaseEstimator
            | TPOTManager
            | Objective
            | InverseObjectives
            | DatasetParams
            | EvolutionParams
            | EvalParams
            | EndParams
            | RuntimeParams
    )
    def _(obj) -> LgConfig:
        return cfg.instance_init_args_to_config(obj)

    @cfg.to_config.register(_Scorer)
    def _(obj: _Scorer) -> LgConfig:
        return cfg.instance_to_config(
            type(obj),
            score_func=obj._score_func,
            sign=obj._sign,
            kwargs=obj._kwargs,
            response_method=obj._response_method,
        )

    # @cfg.to_config.register(Pipeline)
    # def _(obj: Pipeline) -> dict:
    #     method = f"{type(obj).__module__}.{type(obj).__name__}"
    #     return {
    #         "method": method,
    #         "params": obj.get_params(deep=False),
    #     }
    #
    # @cfg.to_config.register(GraphPipeline)
    # def _(obj: GraphPipeline) -> dict:
    #     method = f"{type(obj).__module__}.{type(obj).__name__}"
    #     return {
    #         "method": method,
    #         "params": obj.get_params(deep=False),
    #     }
    #
    # @cfg.to_config.register(DiGraph)
    # def _(obj: DiGraph) -> dict:
    #     return {key: value for key, value in obj.nodes.items()}
    #
    # @cfg.to_config.register(object)
    # def _(obj: object) -> dict | str:
    #     if hasattr(obj, "__dict__"):
    #         method = f"{type(obj).__module__}.{type(obj).__name__}"
    #         return {"method": method, "params": {
    #             key: value
    #             for key, value in obj.__dict__.items()
    #             if not key.startswith("_") and not key.endswith("_")
    #         }}
    #     return repr(obj)
