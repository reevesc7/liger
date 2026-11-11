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
from sklearn import (
    cross_decomposition,
    ensemble,
    kernel_ridge,
    linear_model,
    multioutput,
    pipeline,
)
from tpot import GraphPipeline
from tpot.objectives import complexity


def score_complexity(est: Any, _X: Any = None, _y: Any = None) -> int:
    match est:
        case (
            cross_decomposition.PLSRegression()
            | linear_model.BayesianRidge()
        ):
            n_params = int(complexity._count_nonzero_coefficients_and_intercept(est))
        case (
            ensemble.AdaBoostRegressor()
            | ensemble.AdaBoostClassifier()
            | ensemble.BaggingRegressor()
            | ensemble.BaggingClassifier()
            | multioutput.MultiOutputRegressor()
            | multioutput.MultiOutputClassifier()
        ):
            n_params = sum(score_complexity(estimator) for estimator in est.estimators_)
        case kernel_ridge.KernelRidge():
            n_params = est.dual_coef_.size
        case pipeline.FeatureUnion():
            n_params = sum(
                score_complexity(estimator)
                for _, estimator in est.transformer_list
            )
        case pipeline.Pipeline():
            return sum(score_complexity(estimator) for _, estimator in est.steps)
        case GraphPipeline():
            n_params = sum(
                score_complexity(node["instance"])
                for node in est.graph.nodes.values()
            )
        case _:
            n_params = int(complexity.calculate_model_complexity(est))
    return n_params
