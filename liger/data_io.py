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


import numpy as np
from numpy.typing import ArrayLike
import random
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from .operations import interpolate_points
from .dataset import Dataset


# Returns a point that deviates along all dimensions according to a normal distribution, scaled by "noise".
def noisy_point(point: ArrayLike, noise: float) -> np.ndarray:
    return point + (noise * np.random.standard_normal(point.shape[0]))


# Generates a random dataset from a random point1 and point2, and an interpolated point. Uses equal distribution for endpoint placement and interpolation.
def random_endpts_dataset(n_entries: int, n_dimensions: int = 2, d_range: float = 0.5, average: float = 0.0, noise: float = 0.0, random_state = None) -> Dataset:
    if random_state != None:
        random.seed(random_state)
    dataset = Dataset(n_entries, n_dimensions*3)
    for entry in range(n_entries):
        point1 = [d_range*(random.random()-0.5+average) for dimension in range(n_dimensions)]
        point2 = [d_range*(random.random()-0.5+average) for dimension in range(n_dimensions)]
        alpha = random.random()
        dataset.X[entry] = np.concatenate((point1, point2, noisy_point(interpolate_points(point1, point2, alpha), noise)))
        dataset.y[entry] = alpha*10
    return dataset


# Generates a random dataset from a given point1 and point2, and an interpolated point. Uses equal distribution for interpolation.
def random_scores_dataset(n_entries: int, point1: ArrayLike, point2: ArrayLike, n_dimensions: int = 2, noise: float = 0.0, random_state = None) -> Dataset:
    point1 = np.array(point1)
    point2 = np.array(point2)
    if random_state != None:
        random.seed(random_state)
    dataset = Dataset(n_entries, n_dimensions*3)
    for entry in range(n_entries):
        alpha = random.random()
        dataset.X[entry] = np.concatenate((point1, point2, noisy_point(interpolate_points(point1, point2, alpha), noise)))
        dataset.y[entry] = alpha*10
    return dataset


def plot2D(data: ArrayLike, title: str = None, axis_labels: list[str] = None, trend_orders: list[int] = [], plot_perfect: bool = False, show_plot: bool = False, filename: str = None) -> list[np.poly1d]:
    plt.scatter(data[0], data[1], color="black", alpha=0.3)
    trend_Xvals = [(max(data[0])-min(data[0]))*float(i)/32 + min(data[0]) for i in range(33)]   
    plt.title(title, fontsize='small')
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


def plot_expl_var_ratios(data: ArrayLike, title: str = None, max_components: int = 16, show_plot: bool = False, filename: str = None):
    pca = PCA(n_components=max_components)
    pca.fit(data)
    expl_var_ratios = [sum(pca.explained_variance_ratio_[:n_components]) for n_components in range(max_components)]
    plt.plot(range(max_components), expl_var_ratios, marker='o', color='black')
    plt.title(title, fontsize='small')
    plt.xlabel('n_components')
    plt.ylabel('Explained variance ratio')
    if filename != None:
        plt.savefig(filename)
    if show_plot:
        plt.show()
    plt.close()
    return expl_var_ratios

