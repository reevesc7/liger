import numpy as np
from numpy.typing import ArrayLike


# Calculates a point along the line segment between point1 and point2, positioned in a range [0,1] along the segment.
def interpolate_points(point1: ArrayLike, point2: ArrayLike, alpha: float) -> np.ndarray:
    point1 = np.array(point1)
    point2 = np.array(point2)
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha should be between 0 and 1.")
    interpolated_point = point1 + alpha * (point2 - point1)
    return interpolated_point


# Calculates the projection of AP onto AB as a proportion of the magnitude of AB.
def vector_projection(ABvector: ArrayLike, APvector: ArrayLike) -> float:
    return np.dot(APvector, ABvector) / np.dot(ABvector, ABvector)


# Calculates the rejection of AP from AB.
def vector_rejection(ABvector: ArrayLike, APvector: ArrayLike) -> float:
    AP0vector = vector_projection(ABvector, APvector) * ABvector
    return np.linalg.norm(APvector - AP0vector)

def compute_R2(data: list[ArrayLike], equations: list) -> list:
    actual_y = data[1]
    return [np.corrcoef(actual_y, [equation(x) for x in data[0]])[0,1]**2 for equation in equations]