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


from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def show() -> None:
    plt.show()


def _ndims(data: ArrayLike, depth: int = 0) -> int:
    if isinstance(data, Iterable):
        depths = {_ndims(item, depth + 1) for item in data}
        if len(depths) > 1:
            raise ValueError("Inconsistent dimensionality in data")
        return next(iter(depths))
    return depth


def _fit_to_dim(data: ArrayLike, dim: int) -> np.ndarray | list[np.ndarray]:
    ndims = _ndims(data)
    if ndims < 2 or ndims > 3:
        raise ValueError(f"Data must be 2D or 3D, but was {ndims}D")
    if ndims <= 2:
        data = np.asarray(data)
        if data.shape[0] != dim:
            raise ValueError(f"2D data must be shape ({dim}, n), but was {data.shape}")
        return data
    if not isinstance(data, Iterable):
        raise TypeError("Data must be iterable (i.e., 2D or 3D, not 0D)")
    data = [np.asarray(item) for item in data]
    for item in data:
        if item.shape[0] != dim:
            raise ValueError(f"3D data arrays must be shape ({dim}, n), but was {item.shape}")
    return data


def _1_scatter(ax: Axes, data: np.ndarray, trend_orders: list[int], plot_perfect: bool) -> None:
    ax.scatter(data[0], data[1], alpha=0.3)
    trend_Xvals = [
        (max(data[0]) - min(data[0])) * float(i) / 32 + min(data[0])
        for i in range(33)
    ]
    for trend_order in trend_orders:
        trend_eq = np.poly1d(np.polyfit(data[0], data[1], trend_order))
        ax.plot(trend_Xvals, trend_eq(trend_Xvals))
    if plot_perfect:
        perfect_mxb = np.poly1d([1,0])
        ax.plot(trend_Xvals, perfect_mxb(trend_Xvals), color="gray")


def scatter(
    data: ArrayLike,
    title: str | None = None,
    axis_labels: tuple[str, str] | None = None,
    trend_orders: list[int] = [],
    plot_perfect: bool = False,
) -> Figure:
    """Create a 2D scatter plot.

    Points on the plot are circles with `alpha=0.3`.

    Parameters
    ----------
    `data` : `ArrayLike`
        An array or iterable of arrays of shape `(2, n)` of scalar values,
        where each array represents a dataset and `n` is the number of data points
        in each dataset.
    `title` : `str`, optional
        A title for the plot.
    `axis_labels` : `tuple[str]`, optional
        A labels for the plot's x and y axes.
    `trend_orders` : `list[int]`, default `[]`
        A list of orders of fitted trendlines to plot. Trendlines colors are
        blue, orange, green, red, purple, etc., in the order given, respectively.
    `plot_perfect` : `bool`, default `False`
        Whether to plot an ideal trendline, assuming the dimensions of the data
        are fully covariant, i.e., $Y=X$. If plotted, this line is in gray.

    Returns
    -------
    `fig` : `matplotlib.figure.Figure`
        The figure plotting the data with any trendlines drawn. Show any current
        figures with `liger.plotting.show()`, and save it with `fig.savefig()`.
    """
    data = _fit_to_dim(data, 2)
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title, fontsize='small')
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    if isinstance(data, np.ndarray):
        _1_scatter(ax, data, trend_orders, plot_perfect)
        return fig
    for dataset in data:
        _1_scatter(ax, dataset, trend_orders, plot_perfect)
    return fig


def plot(
    data: ArrayLike,
    title: str | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> Figure:
    data = _fit_to_dim(data, 2)
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(title, fontsize='small')
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
    if isinstance(data, np.ndarray):
        ax.plot(data[0], data[1])
        return fig
    for dataset in data:
        ax.plot(dataset[0], dataset[1])
    return fig

