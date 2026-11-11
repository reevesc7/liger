from enum import StrEnum
from typing import Self
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray


TOL = 2 ** -32


class KneeStatus(StrEnum):
    OFF_PLANE = "off_plane"
    ON_PLANE = "on_plane"
    DOMINANT = "dominant"
    DEGENERATE = "degenerate"


class HyperplaneError(Exception):
    pass


@dataclass(slots=True, frozen=True)
class Hyperplane:
    normal: NDArray[np.float64]
    offset: np.float64

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        points = np.asarray(points, dtype=np.float64)
        edges = points[1:] - points[0]
        _, s, vh = np.linalg.svd(edges)
        if s[-1] <= TOL * s[0]:
            raise HyperplaneError("Degenerate (rank-deficient) points; "
                "no unique hyperplane")
        normal = vh[-1]
        return cls(normal, -normal @ points[0])

    def oriented_toward(self, point: ArrayLike) -> Self:
        distance = np.float64(self.distances(point))
        if abs(distance) <= TOL:
            raise HyperplaneError("Reference point lies on the hyperplane")
        return self if distance > 0.0 else type(self)(-self.normal, -self.offset)

    def distances(self, points: ArrayLike) -> NDArray[np.float64]:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.normal + self.offset


@dataclass(slots=True, frozen=True)
class KneeResult:
    indices: NDArray[np.intp]
    status: KneeStatus
    max_distance: np.float64 | None = None

    def unique_knee_indices(self, points: ArrayLike) -> NDArray[np.intp]:
        points = np.asarray(points)
        return np.unique(points[self.indices], return_index=True, axis=0)[1]


class KneeUndefinedError(Exception):
    def __init__(self, knee_result: KneeResult) -> None:
        super().__init__("Pareto knee is undefined: "
            f"{knee_result.indices.size} knees, with {knee_result.status} status")
        self.result = knee_result


def _validate(points: NDArray[np.float64], *, empty_ok: bool = True) -> None:
    if points.ndim != 2:
        raise ValueError(f"points must be 2D but were shape {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Some values within points are non-finite")
    if not empty_ok and points.shape[0] == 0:
        raise ValueError("Received empty points array")


def _is_dominated(point: NDArray[np.float64], others: NDArray[np.float64]) -> np.bool_:
    return ((others >= point).all(axis=1) & (others > point).any(axis=1)).any()


def pareto_mask(points: ArrayLike) -> NDArray[np.bool_]:
    points = np.asarray(points, dtype=np.float64)
    _validate(points)
    n = points.shape[0]
    mask = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return mask
    order = np.lexsort(points.T[::-1])[::-1]
    front = np.empty_like(points)
    front_index = 0
    for index in order:
        point = points[index]
        cur_front = front[:front_index]
        if front_index != 0 and _is_dominated(point, cur_front):
            continue
        front[front_index] = point
        front_index += 1
        mask[index] = True
    return mask


def pareto_indices(points: ArrayLike) -> NDArray[np.intp]:
    return np.flatnonzero(pareto_mask(points))


def extremes_indices(points: ArrayLike) -> NDArray[np.intp]:
    points = np.asarray(points, dtype=np.float64)
    indices = np.empty(points.shape[1], dtype=np.intp)
    for dim in range(points.shape[1]):
        best_xs = np.flatnonzero(points[:, dim] == points[:, dim].max())
        sum_of_other_dims = points[best_xs].sum(axis=1) - points[best_xs, dim]
        indices[dim] = best_xs[sum_of_other_dims.argmax()]
    return indices


def pareto_knee(points: ArrayLike) -> KneeResult:
    """Determine the index or indices of the Pareto knee or knee candidates
    in a set of multi-objective points.

    Objectives are maximized, i.e., greater values are treated as better.
    The knee point is the point farthest from and on the optimal side of
    the hyperplane formed by the extreme points within each objective.

    Parameters
    ----------
    points : ArrayLike
        2D array of points from which to draw the knee. Points must contain only
        finite values.

    Returns
    -------
    knee_result : KneeResult
        Potential knee(s), including a list of indices and a status (type KneeStatus)
        denoting whether the knee(s) is/are located off or on the hyperplane,
        the knee(s) is/are dominant, or that extreme points are degenerate.
        - If there are multiple dominant knees, they have the exact same scores.
        - If there are multiple non-dominant knees, they either have the exact same
          scores or are the same distance from the hyperplane. In the case of on-plane
          knees, this distance ~= 0.0.
        - If extreme points are degenerate, knees are the Pareto front of points.
    """
    points = np.asarray(points, dtype=np.float64)
    _validate(points, empty_ok=False)
    extreme_xs = extremes_indices(points)
    unique_extremes = np.unique(points[extreme_xs], axis=0)
    if unique_extremes.shape[0] == 1:
        best_xs = np.flatnonzero((points == unique_extremes[0]).all(axis=1))
        return KneeResult(best_xs, KneeStatus.DOMINANT)
    ideal = points.max(axis=0)
    try:
        plane = Hyperplane.from_points(points[extreme_xs]).oriented_toward(ideal)
    except HyperplaneError:
        return KneeResult(pareto_indices(points), KneeStatus.DEGENERATE)
    if (plane.normal < -TOL).any():
        valid_xs = pareto_indices(points)
    else:
        valid_xs = np.arange(points.shape[0])
    distances = plane.distances(points[valid_xs])
    max_distance = distances.max()
    best_xs = np.flatnonzero(distances >= max_distance - TOL)
    if best_xs.size > 1:
        best_xs = best_xs[pareto_mask(points[valid_xs[best_xs]])]
    if max_distance <= TOL:
        return KneeResult(valid_xs[best_xs], KneeStatus.ON_PLANE, max_distance)
    return KneeResult(valid_xs[best_xs], KneeStatus.OFF_PLANE, max_distance)


def pareto_knee_index(points: ArrayLike) -> int:
    """A strict form of pareto_knee(), yielding a single index.

    Raises if the knee candidates cannot be narrowed to a single point.
    The following cases may result in multiple knee candidates.

    Extreme points are degenerate. In such cases, 2 or more objectives
    may be highly correlated or an objective may have no score variation.
    There are infinitely may admissible extreme hyperplanes, potentially
    selecting different knees. The Pareto front is a superset of viable knees.

    No points lie on the optimal side of the extreme hyperplane.
    In such cases, the Pareto front may be linear or bowed behind the hyperplane.
    Optimal points are arbitrarily any point on the hyperplane, including extremes.

    The maximum distance from the extreme hyperplane is shared by multiple points.

    Parameters
    ----------
    points : ArrayLike
        2D array of points from which to draw the knee. Points must contain only
        finite values.

    Returns
    -------
    knee_index : int
        Index of the knee point
    """
    result = pareto_knee(points)
    if result.indices.size == 1:
        return int(result.indices[0])
    raise KneeUndefinedError(result)
