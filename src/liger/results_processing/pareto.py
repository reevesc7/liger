from typing import Self
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd


@dataclass(slots=True)
class Hyperplane:
    normal: np.ndarray
    offset: np.ndarray

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        points = np.asarray(points)
        edges = points[1:] - points[0]
        _, _, null_space = np.linalg.svd(edges)
        normal = np.asarray(null_space[-1])
        offset = np.asarray(-normal @ points[0])
        return cls(normal, offset)

    def distances(self, points: ArrayLike) -> np.ndarray:
        points = np.asarray(points)
        return points @ self.normal + self.offset


def pareto_knee(
    pareto_scores: pd.DataFrame,
    objectives_weights: ArrayLike,
) -> pd.Series:
    objectives_weights = np.asarray(objectives_weights)
    if objectives_weights.ndim != 1:
        raise ValueError(
            "objectives_weights must be 1D "
            f"but is {objectives_weights.shape} "
            f"(should be ({pareto_scores.shape[1]},)"
        )
    if objectives_weights.size != pareto_scores.shape[1]:
        raise ValueError(
            "objectives_weights must be the same size as pareto_scores columns "
            f"but is {objectives_weights.shape} "
            f"(should be ({pareto_scores.shape[1]},)"
        )
    if pareto_scores.shape[0] == 1:
        return pareto_scores.iloc[0]
    pareto_array = np.asarray(pareto_scores)
    maxed_pareto = pareto_array * objectives_weights
    extreme_indices = np.argmax(maxed_pareto, axis=0)
    if pareto_scores.shape[1] > 1 and (
        not isinstance(extreme_indices, np.ndarray)
        or np.unique(extreme_indices.size) < extreme_indices.size
    ):
        raise ValueError(
            "Data contains degenerate extreme points; could not determine knee point. "
            "Two or more objectives may be highly correlated with one another, "
            "or all solutions may have the same score for one or more objectives."
        )
    extremes = pareto_array[extreme_indices]
    hyperplane = Hyperplane.from_points(extremes)
    distances = hyperplane.distances(pareto_scores)
    knee = pd.Series(pareto_scores.iloc[np.abs(distances).argmax()])
    return knee
