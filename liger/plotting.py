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


from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def show() -> None:
    plt.show()


def _ensure_same_shape(arrays: Sequence[np.ndarray]) -> None:
    if len(arrays) == 0:
        return
    shape_1 = arrays[0].shape
    for array in arrays[1:]:
        if array.shape != shape_1:
            raise ValueError("Inconsistent dimensionality in data")


def _set_titles(ax: Axes, title: str | None, axis_labels: tuple[str, str] | None) -> None:
    if title is not None:
        ax.set_title(title, fontsize='small')
    if axis_labels is not None:
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])


def _single_scatter(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    trend_orders: list[int],
    plot_perfect: bool
) -> None:
    ax.scatter(x, y, alpha=0.3)
    trend_Xvals = [
        (max(x) - min(x)) * float(i) / 32 + min(x)
        for i in range(33)
    ]
    for trend_order in trend_orders:
        trend_eq = np.poly1d(np.polyfit(x, y, trend_order))
        ax.plot(trend_Xvals, trend_eq(trend_Xvals))
    if plot_perfect:
        perfect_mxb = np.poly1d([1,0])
        ax.plot(trend_Xvals, perfect_mxb(trend_Xvals), color="gray")


def scatter(
    x: ArrayLike,
    y: ArrayLike,
    title: str | None = None,
    axis_labels: tuple[str, str] | None = None,
    trend_orders: list[int] = [],
    plot_perfect: bool = False,
) -> Figure:
    """Create a 2D scatter plot.

    Points on the plot are circles with `alpha=0.3`.

    Parameters
    ----------
    `x` : `ArrayLike`
        A 1- or 2-dimensional array-like, of x-values from one or more datasets.
    `y` : `ArrayLike`
        A 1- or 2-dimensional array-like, of y-values from one or more datasets.
        Must be the same shape as `x`.
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
    x = np.asarray(x)
    y = np.asarray(y)
    _ensure_same_shape((x, y))
    fig, ax = plt.subplots()
    _set_titles(ax, title, axis_labels)
    if len(x.shape) == 1:
        _single_scatter(ax, x, y, trend_orders, plot_perfect)
        return fig
    for dataset in range(x.shape[0]):
        _single_scatter(ax, x[dataset], y[dataset], trend_orders, plot_perfect)
    return fig


def plot(
    x: ArrayLike,
    y: ArrayLike,
    title: str | None = None,
    axis_labels: tuple[str, str] | None = None,
) -> Figure:
    """Create a 2D line plot.

    Parameters
    ----------
    `x` : `ArrayLike`
        A 1- or 2-dimensional array-like, of x-values from one or more datasets.
    `y` : `ArrayLike`
        A 1- or 2-dimensional array-like, of y-values from one or more datasets.
        Must be the same shape as `x`.
    `title` : `str`, optional
        A title for the plot.
    `axis_labels` : `tuple[str]`, optional
        A labels for the plot's x and y axes.

    Returns
    -------
    `fig` : `matplotlib.figure.Figure`
        The figure plotting the data. Show any current
        figures with `liger.plotting.show()`, and save it with `fig.savefig()`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    _ensure_same_shape((x, y))
    fig, ax = plt.subplots()
    _set_titles(ax, title, axis_labels)
    if len(x.shape) == 1:
        ax.plot(x, y)
        return fig
    for dataset in range(x.shape[0]):
        ax.plot(x[dataset], y[dataset])
    return fig

