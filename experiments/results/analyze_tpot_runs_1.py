from typing import Any, Iterable
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
from liger.results_processing.tpot.load_data import mass_load_data, load_data


def plot_2d_scores(indiv_data: pd.DataFrame) -> None:
    # indiv_data["complexity_scorer"] = indiv_data["complexity_scorer"].clip(0, 2_000)
    # indiv_data["score_softmax_wasserstein"] = indiv_data["score_softmax_wasserstein"].clip(-0.125, 0)
    indiv_data["complexity_scorer"] = indiv_data["complexity_scorer"]
    indiv_data["score_softmax_wasserstein"] = indiv_data["score_softmax_wasserstein"]
    nonpareto: pd.DataFrame = indiv_data.loc[indiv_data["Pareto_Front"] != 1].iloc[:, :2]
    pareto: pd.DataFrame = indiv_data.loc[indiv_data["Pareto_Front"] == 1].iloc[:, :2]
    # lgplt.scatter(
    #     (nonpareto.transpose(), pareto.transpose()),
    #     "test_pareto",
    #     ("W dist", "complexity"),
    # ).savefig("test_pareto")
    fig, ax = plt.subplots()
    ax.set_title("test_pareto")
    ax.set_xlabel("W dist")
    ax.set_ylabel("complexity")
    print(pareto.iloc[:, 0])
    ax.scatter(pareto.iloc[:, 0], pareto.iloc[:, 1], alpha=0.5)
    ax.scatter(nonpareto.iloc[:, 0], nonpareto.iloc[:, 1], alpha=0.5)
    ax.autoscale(False)
    fig.savefig("test_pareto")


def retrieve_data() -> dict[str, Any] | None:
    # return load_data("../../../runs_data/e02/outputs/e02/2026-02-19_20-19-51.516349/")
    # return load_data("../tpot/outputs/testing/2026-04-09_01-35-43.898209/")
    return next((data for data in mass_load_data(
        "../../../runs_data/e02/outputs/e02/2026-02-19_20-19-51.516349/",
        "*",
    )), None)


def _get_pareto_scores(eval_indivs: pd.DataFrame, n_objectives: int) -> pd.DataFrame:
    return eval_indivs.loc[eval_indivs["Pareto_Front"] == 1].iloc[:, :n_objectives]


def plot_paretos(runs_data: Iterable[dict[str, Any]]) -> None:
    fig, ax = plt.subplots()
    ax.set_title("test_pareto")
    ax.set_xlabel("W dist")
    ax.set_ylabel("log(complexity)")
    for run_data in runs_data:
        print(run_data["manager_parameters"]["id"])
        pareto_scores = _get_pareto_scores(run_data)
        ax.scatter(pareto_scores.iloc[:, 0], np.log(pareto_scores.iloc[:, 1]), alpha = 0.5)
    fig.savefig("test_paretos")


@dataclass(slots=True)
class Hyperplane:
    normal: np.ndarray
    offset: np.ndarray

    def distances(self, points: ArrayLike) -> np.ndarray:
        points = np.asarray(points)
        return points @ self.normal + self.offset


def _define_hyperplane(points: ArrayLike) -> Hyperplane:
    points = np.asarray(points)
    edges = points[1:] - points[0]
    _, _, null_space = np.linalg.svd(edges)
    normal = np.asarray(null_space[-1])
    offset = np.asarray(-normal @ points[0])
    return Hyperplane(normal, offset)


def _dist_from_line(row: pd.Series, origin: pd.Series, line: pd.Series) -> np.floating:
    line3 = np.array([line.iloc[0], line.iloc[1], 0.0])
    vec = row - origin
    vec3 = np.array([vec.iloc[0], vec.iloc[1], 0.0])
    return np.linalg.norm(np.cross(line3, vec3)) / np.linalg.norm(line)


def _format_knee(knee: pd.Series, id: str) -> pd.Series:
    knee["Individual ID"] = knee.name
    knee.name = id
    return knee


def _pareto_knee(
    pareto_scores: pd.DataFrame,
    objectives_weights: ArrayLike,
    id: str,
) -> pd.Series | None:
    objectives_weights = np.asarray(objectives_weights)
    if objectives_weights.ndim != 1 or objectives_weights.size != pareto_scores.shape[1]:
        print(
            "objectives_weights must be 1D and the same size as pareto_scores columns "
            f"but is {objectives_weights.shape} "
            f"(should be ({pareto_scores.shape[1]},)"
        )
        return None
    if pareto_scores.shape[0] == 1:
        return _format_knee(pareto_scores.iloc[0], id)
    pareto_array = np.asarray(pareto_scores)
    maxed_pareto = pareto_array * objectives_weights
    extreme_indices = np.argmax(maxed_pareto, axis=0)
    # print(maxed_pareto)
    # print(extreme_indices)
    if len(set(extreme_indices)) < len(extreme_indices):
        print(
            f"Degenerate extreme points found in {id}; could not determine knee point. "
            "Two or more score categories may be highly correlated with one another."
        )
        return None
    extremes = pareto_array[extreme_indices]
    hyperplane = _define_hyperplane(extremes)
    distances = hyperplane.distances(pareto_scores)
    # print(distances)
    # print(np.abs(distances).argmax())
    knee = pd.Series(pareto_scores.iloc[np.abs(distances).argmax()])
    # sorted = pareto.sort_values(str(pareto.columns[0]))
    # leftmost = sorted.iloc[0, :]
    # print(leftmost)
    # print(sorted.iloc[-1, :])
    # span_line = sorted.iloc[-1, :] - leftmost
    # sorted["_dist"] = sorted.apply(lambda row: _dist_from_line(row, leftmost, span_line), axis=1)
    # # print(sorted)
    # knee: pd.Series = sorted.sort_values("_dist").drop("_dist", axis=1).iloc[-1, :]
    return _format_knee(knee, id)


def plot_knees(runs_data: Iterable[dict[str, Any]]) -> None:
    knees = []
    # normalized_knees = []
    for run_data in runs_data:
        print(run_data["manager_parameters"]["id"])
        evaluated_individuals = run_data["tpot_attributes"]["evaluated_individuals"]
        scorers_weights = run_data["tpot_parameters"]["scorers_weights"]
        other_objectives_weights = run_data["tpot_parameters"]["other_objective_functions_weights"]
        objectives_weights = scorers_weights + other_objectives_weights
        pareto_scores = _get_pareto_scores(evaluated_individuals, len(objectives_weights))
        knee = _pareto_knee(
            pareto_scores,
            objectives_weights,
            run_data["manager_parameters"]["id"],
        )
        if knee is not None:
            knees.append(knee)
        elif len(objectives_weights) == 1:
            print("Only 1 objective; selecting top scorer")
            knees.append(pareto_scores.multiply(
                objectives_weights,
                axis=1,
            )[pareto_scores.columns[0]].idxmax())
        # normalized_pareto_scores = pareto_scores.assign(
        #     complexity_scorer=pareto_scores["complexity_scorer"] / 175_000_000_000,
        # )
        # normalized_knees.append(_pareto_knee(normalized_pareto_scores, run_data["manager_parameters"]["id"]))
    knees = pd.DataFrame(knees)
    print(knees)
    # print(pd.DataFrame(normalized_knees))
    fig, ax = plt.subplots()
    ax.set_title("test_knees")
    ax.set_xlabel("W dist")
    ax.set_ylabel("complexity")
    ax.scatter(knees.iloc[:, 0], knees.iloc[:, 1], alpha = 0.5)
    fig.savefig("e03/w--/test_knees")


def main():
    # data = retrieve_data()
    # if data is None:
    #     print("No data")
    #     return
    # plot_2d_scores(data["tpot_attributes"]["evaluated_individuals"])
    data = mass_load_data(
        "../../../runs_data/e03/outputs/e03.0",
        "*",
        filters=[
            {
                "manager_parameters": {
                    "data_file": "data/smallville_417_maria_embprop.csv",
                    "feature_keys": [
                        "all-mpnet-base-v2_--e"
                    ],
                }
            }
        ],
    )
    # plot_paretos(data)
    plot_knees(data)


if __name__ == "__main__":
    main()

