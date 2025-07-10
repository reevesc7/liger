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


from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from .dataset import Dataset


def plot2D(
    data: np.ndarray,
    title: str | None = None,
    axis_labels: list[str] | None = None,
    trend_orders: list[int] = [],
    plot_perfect: bool = False,
    show_plot: bool = False,
    filename: str | Path | None = None,
) -> list[np.poly1d]:
    plt.scatter(data[0], data[1], color="black", alpha=0.3)
    trend_Xvals = [(max(data[0])-min(data[0]))*float(i)/32 + min(data[0]) for i in range(33)]   
    if title is not None:
        plt.title(title, fontsize='small')
    if axis_labels is not None:
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    #plot trendlines
    trend_eqs = []
    for trend_order in trend_orders:
        trend_eq = np.poly1d(np.polyfit(data[0], data[1], trend_order))
        plt.plot(trend_Xvals, trend_eq(trend_Xvals))
        trend_eqs.append(trend_eq)
    if plot_perfect:
        perfect_mxb = np.poly1d([1,0])
        plt.plot(trend_Xvals, perfect_mxb(trend_Xvals), color="gray")
    if filename != None:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    return trend_eqs


def plot_expl_var_ratios(
    data: np.ndarray,
    title: str | None = None,
    max_components: int = 16,
    show_plot: bool = False,
    filename: str | Path | None = None,
):
    pca = PCA(n_components=max_components)
    pca.fit(data)
    expl_var_ratios = [sum(pca.explained_variance_ratio_[:n_components]) for n_components in range(max_components)]
    plt.plot(range(max_components), expl_var_ratios, marker='o', color='black')
    if title is not None:
        plt.title(title, fontsize='small')
    plt.xlabel('n_components')
    plt.ylabel('Explained variance ratio')
    if filename != None:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    return expl_var_ratios

