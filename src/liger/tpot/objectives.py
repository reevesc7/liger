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


LINEAR = (
    cross_decomposition.PLSRegression,
    linear_model.BayesianRidge,
)
WRAPPER = (
    ensemble.AdaBoostRegressor,
    ensemble.AdaBoostClassifier,
    ensemble.BaggingRegressor,
    ensemble.BaggingClassifier,
    multioutput.MultiOutputRegressor,
    multioutput.MultiOutputClassifier,
)


def score_complexity(est: Any, _X: Any = None, _y: Any = None) -> int:
    if isinstance(est, LINEAR):
        n_params = int(complexity._count_nonzero_coefficients_and_intercept(est))
    elif isinstance(est, WRAPPER):
        n_params = sum(score_complexity(estimator) for estimator in est.estimators_)
    elif isinstance(est, kernel_ridge.KernelRidge):
        n_params = est.dual_coef_.size
    elif isinstance(est, pipeline.FeatureUnion):
        n_params = sum(
            score_complexity(estimator)
            for _, estimator in est.transformer_list
        )
    elif isinstance(est, pipeline.Pipeline):
        n_params = sum(score_complexity(estimator) for _, estimator in est.steps)
    elif isinstance(est, GraphPipeline):
        n_params = sum(
            score_complexity(node["instance"])
            for node in est.graph.nodes.values()
        )
    else:
        n_params = int(complexity.calculate_model_complexity(est))
    return n_params
