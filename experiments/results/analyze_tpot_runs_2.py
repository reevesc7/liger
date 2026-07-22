from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from cycler import cycler
from liger.results_processing.json_cache import JSONCache
from liger.results_processing.pareto import pareto_knee
from liger.results_processing.tpot.demographics import TPOTDemographics


INPUT_DIR = Path("../../../runs_data/e03/outputs/e03.0")
OUTPUT_DIR = Path("e03/w--")
# INPUT_DIR = Path("../../../runs_data/e03/outputs/e03.1")
# OUTPUT_DIR = Path("e03/wc-")
# INPUT_DIR = Path("../../../runs_data/e03/outputs/e03.2")
# OUTPUT_DIR = Path("e03/w-n")


# cache = JSONCache(
#     "../../../runs_data/e03/outputs/e03.0",
#     "*",
# )
# demog = DemographicsAnalyzer(cache)
# print("\nINDIVIDUALS")
# print(demog.individuals)
# print(demog.individuals.columns)
# print("\nCONNECTIONS")
# print(demog.connections)
# print(demog.connections.columns)
# print("\nN_NODES")
# print(demog.n_nodes)
# print("\nN_BRANCHES")
# print(demog.n_branches)
# print("\nTOTAL")
# total = demog.individuals.join((demog.n_nodes, demog.n_branches))
# print(total)
# print(total.columns)
# print("\nMETHOD")
# print(demog.n_nodes_of_method)
# print(demog.individuals.join(demog.n_nodes_of_method))
# print("\nLEAF METHOD")
# print(demog.n_leaves_of_method)
# print("\nROOTS")
# print(demog.root_methods)


plt.rc("axes", prop_cycle=cycler(linestyle=["-", "--", "-."]) * cycler(color=[
    color["color"]
    for color in mpl.rcParams["axes.prop_cycle"]
]))


def plot_n_nodes(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("number of nodes per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of nodes (mean, SEM)")
    gen_run_n_nodes = demog.individuals.join(demog.n_nodes).groupby([
        "run_id",
        "Generation",
    ])
    mean_n_nodes = gen_run_n_nodes["n_nodes"].mean()
    sem_n_nodes = gen_run_n_nodes["n_nodes"].sem()
    for run_id in mean_n_nodes.index.get_level_values("run_id").unique():
        run_means = mean_n_nodes.xs(run_id, level="run_id")
        run_sems = sem_n_nodes.xs(run_id, level="run_id")
        ax.plot(run_means.index, run_means)
        ax.fill_between(
            run_sems.index,
            run_means - run_sems,
            run_means + run_sems,
            alpha=0.2,
        )


def plot_n_branches(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("number of branches per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of branches (mean, SEM)")
    # gen_run_n_branches = demog.individuals.join(demog.n_branches).groupby([
    #     "run_id",
    #     "Generation",
    # ])
    gen_run_n_branches = demog.n_branches.groupby(["run_id", "Generation"])
    mean_n_branches = gen_run_n_branches.mean()
    sem_n_branches = gen_run_n_branches.sem()
    for run_id in mean_n_branches.index.get_level_values("run_id").unique():
        run_means = mean_n_branches.xs(run_id, level="run_id")
        run_sems = sem_n_branches.xs(run_id, level="run_id")
        ax.plot(run_means.index, run_means)
        ax.fill_between(
            run_sems.index,
            run_means - run_sems,
            run_means + run_sems,
            alpha=0.2,
        )


def plot_eval_times(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("evaluation time per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("evaluation time (s) (mean, SEM)")
    gen_run_eval_times = pd.Series(
        demog.individuals["Completed Timestamp"] - demog.individuals["Submitted Timestamp"]
    ).groupby(["run_id", "Generation"])
    mean_eval_times = gen_run_eval_times.mean()
    sem_eval_times = gen_run_eval_times.sem()
    for run_id in mean_eval_times.index.get_level_values("run_id").unique():
        run_means = mean_eval_times.xs(run_id, level="run_id")
        run_sems = sem_eval_times.xs(run_id, level="run_id")
        ax.plot(run_means.index, run_means)
        ax.fill_between(
            run_sems.index,
            run_means - run_sems,
            run_means + run_sems,
            alpha=0.2,
        )


def plot_pareto_scores(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("Pareto front scores, by run")
    scores = pd.DataFrame(
        demog.individuals.loc[demog.individuals["Pareto_Front"] == 1].drop([
            "Parents",
            "Variation_Function",
            "Submitted Timestamp",
            "Completed Timestamp",
            "Eval Error",
            "Pareto_Front",
        ], axis=1),
    )
    if scores.shape[1] > 2:
        print("Scores are greater than 2 dimensions; skipping Pareto front scores plot")
        return
    elif scores.shape[1] == 1:
        ax.set_xlabel("run")
        ax.set_ylabel(str(scores.columns[0]))
        for index, run_id in enumerate(
            scores.index.get_level_values("run_id").unique()
        ):
            ax.bar(index, scores.xs(run_id, level="run_id").iloc[0, 0])
        return
    ax.set_xlabel(str(scores.columns[0]))
    ax.set_ylabel(str(scores.columns[1]))
    for run_id in scores.index.get_level_values("run_id").unique():
        run_scores = scores.xs(run_id, level="run_id")
        ax.scatter(run_scores.iloc[:, 0], run_scores.iloc[:, 1], alpha=0.2)
    # ax.set_xbound(-0.11, -0.08)
    # ax.set_ybound(-100, 15000)


def plot_pareto_nodes_branches(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("Pareto front number of nodes and branches, by run")
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("number of branches")
    filter = demog.individuals["Pareto_Front"] == 1
    # individuals = pd.DataFrame(demog.individuals.join([
    #     demog.n_nodes,
    #     demog.n_branches,
    # ]).loc[demog.individuals["Pareto_Front"] == 1])
    nodes_branches = pd.concat([demog.n_nodes, demog.n_branches], axis=1)[filter]
    for run_id in nodes_branches.index.get_level_values("run_id").unique():
        run_individuals = nodes_branches.xs(run_id, level="run_id")
        ax.scatter(run_individuals["n_nodes"], run_individuals["n_branches"], alpha=0.2)


def plot_pareto_knee_scores(ax: Axes, demog: TPOTDemographics) -> None:
    scores = pd.DataFrame(
        demog.individuals.loc[demog.individuals["Pareto_Front"] == 1].drop([
            "Parents",
            "Variation_Function",
            "Submitted Timestamp",
            "Completed Timestamp",
            "Eval Error",
            "Pareto_Front",
        ], axis=1),
    )
    if scores.shape[1] > 2:
        print("Scores are greater than 2 dimensions; skipping Pareto knee scores plot")
        return
    elif scores.shape[1] == 1:
        plot_pareto_scores(ax, demog)
        return
    ax.set_title("Pareto knee scores (1 per run)")
    ax.set_xlabel(str(scores.columns[0]) + " +/- SEM")
    ax.set_ylabel(str(scores.columns[1]) + " +/- SEM")
    knees = []
    for run_id in scores.index.get_level_values("run_id").unique():
        run_scores = scores.xs(run_id, level="run_id")
        run = demog.runs.loc[run_id]
        run_objectives_weights = (
            run["scorers_weights"] + run["other_objective_functions_weights"]
        )
        knees.append(pareto_knee(run_scores, run_objectives_weights))
        # run_knee = pareto_knee(run_scores, run_objectives_weights)
        # ax.scatter(run_knee.iloc[0], run_knee.iloc[1], alpha=0.2)
    knees = pd.concat(knees, axis=1).T
    ax.scatter(knees.iloc[:, 0], knees.iloc[:, 1], alpha=0.5)
    xmin = knees.iloc[:, 0].min()
    xmax = knees.iloc[:, 0].max()
    xmean = knees.iloc[:, 0].mean()
    xsem = knees.iloc[:, 0].sem()
    ymin = knees.iloc[:, 1].min()
    ymax = knees.iloc[:, 1].max()
    ymean = knees.iloc[:, 1].mean()
    ysem = knees.iloc[:, 1].sem()
    ax.vlines(xmean, ymin, ymax)
    ax.axvspan(xmean - xsem, xmean + xsem, alpha=0.1)
    ax.hlines(ymean, xmin, xmax)
    ax.axhspan(ymean - ysem, ymean + ysem, alpha=0.1)


def plot_node_methods(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("nodes of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("nodes per individual (mean)")
    n_individuals = demog.individuals.groupby("Generation").size()
    # n_methods = demog.individuals.join(demog.n_nodes_of_method).groupby([
    #     "Generation",
    #     "method",
    # ])["n_nodes_of_method"].sum()
    n_methods = demog.n_nodes_of_method.groupby(["Generation", "method"]).sum()
    full_index = pd.MultiIndex.from_product([
        n_methods.index.get_level_values("Generation").unique(),
        n_methods.index.get_level_values("method").unique(),
    ])
    n_methods = n_methods.reindex(full_index, fill_value=0)
    for method in n_methods.index.get_level_values("method").unique():
        n_method = n_methods.xs(method, level="method")
        ax.plot(
            n_method.index,
            n_method / n_individuals,
            label=str(method).split(".")[-1],
        )
    ax.legend(loc="upper left", framealpha=0)


def plot_root_methods(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("roots of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("roots per individual (mean)")
    n_individuals = demog.individuals.groupby("Generation").size()
    # n_methods = demog.individuals.join(demog.root_methods).groupby([
    #     "Generation",
    #     "root_node",
    # ]).size()
    n_methods = demog.root_methods.groupby(["Generation", demog.root_methods]).size()
    full_index = pd.MultiIndex.from_product([
        n_methods.index.get_level_values("Generation").unique(),
        n_methods.index.get_level_values("root_node").unique(),
    ])
    n_methods = n_methods.reindex(full_index, fill_value=0)
    for method in n_methods.index.get_level_values("root_node").unique():
        n_method = n_methods.xs(method, level="root_node")
        ax.plot(
            n_method.index,
            n_method / n_individuals,
            label=str(method).split(".")[-1],
        )
    ax.legend(loc="upper left", framealpha=0)


def plot_leaf_methods(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("leaves of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("leaves per individual (mean)")
    n_individuals = demog.individuals.groupby("Generation").size()
    # n_methods = demog.individuals.join(demog.n_leaves_of_method).groupby([
    #     "Generation",
    #     "method",
    # ])["n_leaves_of_method"].sum()
    n_methods = demog.n_leaves_of_method.groupby(["Generation", "method"]).sum()
    full_index = pd.MultiIndex.from_product([
        n_methods.index.get_level_values("Generation").unique(),
        n_methods.index.get_level_values("method").unique(),
    ])
    n_methods = n_methods.reindex(full_index, fill_value=0)
    for method in n_methods.index.get_level_values("method").unique():
        n_method = n_methods.xs(method, level="method")
        ax.plot(
            n_method.index,
            n_method / n_individuals,
            label=str(method).split(".")[-1],
        )
    ax.legend(loc="upper left", framealpha=0)


def plot_pareto_root_methods(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("roots of method per Pareto individual, by method")
    ax.set_xlabel("run")
    ax.set_ylabel("roots per individual (mean)")
    filter = demog.individuals["Pareto_Front"] == 1
    n_individuals = pd.DataFrame(demog.individuals.loc[filter]).groupby("run_id").size()
    # n_methods = pd.Series(
    #     demog.individuals.join(demog.root_methods).loc[filter].groupby([
    #         "run_id",
    #         "root_node",
    #     ]).size()
    # )
    n_methods = pd.Series(demog.root_methods.loc[filter]).groupby([
        "run_id",
        demog.root_methods,
    ]).size()
    full_index = pd.MultiIndex.from_product([
        n_methods.index.get_level_values("run_id").unique(),
        pd.Series(demog.root_methods.loc[filter]).unique(),
    ], names=["run_id", "root_node"])
    n_methods = n_methods.reindex(full_index, fill_value=0)
    bottom = np.zeros(n_methods.index.get_level_values("run_id").nunique())
    x = np.arange(bottom.size)
    for method in pd.Series(demog.root_methods.loc[filter]).unique():
        p_method = pd.Series(n_methods.xs(method, level="root_node") / n_individuals)
        ax.bar(x, p_method, bottom=bottom, label=str(method).split(".")[-1])
        bottom += p_method
        # ax.plot(p_method.index, p_method, label=str(method).split(".")[-1])
    ax.legend(loc="upper left", framealpha=0)


def plot_pareto_leaf_methods(ax: Axes, demog: TPOTDemographics) -> None:
    ax.set_title("leaves of method per Pareto individual, by method")
    ax.set_xlabel("run")
    ax.set_ylabel("leaves per individual (mean)")
    filter = demog.individuals["Pareto_Front"] == 1
    n_individuals = pd.DataFrame(demog.individuals.loc[filter]).groupby("run_id").size()
    # n_methods = pd.Series(
    #     demog.individuals.join(demog.n_leaves_of_method).loc[filter].groupby([
    #         "run_id",
    #         "method",
    #     ])["n_leaves_of_method"].sum()
    # )
    n_methods = pd.Series(demog.n_leaves_of_method.loc[filter]).groupby([
        "run_id",
        "method",
    ]).sum()
    full_index = pd.MultiIndex.from_product([
        n_methods.index.get_level_values("run_id").unique(),
        n_methods.index.get_level_values("method").unique(),
    ])
    n_methods = n_methods.reindex(full_index, fill_value=0)
    bottom = np.zeros(n_methods.index.get_level_values("run_id").nunique())
    x = np.arange(bottom.size)
    for method in n_methods.index.get_level_values("method").unique():
        p_method = pd.Series(n_methods.xs(method, level="method") / n_individuals)
        ax.bar(x, p_method, bottom=bottom, label=str(method).split(".")[-1])
        bottom += p_method
        # ax.plot(p_method.index, p_method, label=str(method).split(".")[-1])
    ax.legend(loc="upper left", framealpha=0)


def main():
    cache = JSONCache(
        INPUT_DIR,
        "run_data.json",
    )
    demog = TPOTDemographics(cache)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(40,20))
    plot_n_nodes(ax[0, 0], demog)
    # plot_n_branches(ax[0, 1], demog)
    plot_eval_times(ax[0, 1], demog)
    plot_pareto_scores(ax[0, 2], demog)
    # plot_pareto_nodes_branches(ax[0, 3], demog)
    plot_pareto_knee_scores(ax[0, 3], demog)
    # plot_node_methods(ax[1, 0], demog)
    plot_root_methods(ax[1, 0], demog)
    plot_leaf_methods(ax[1, 1], demog)
    plot_pareto_root_methods(ax[1, 2], demog)
    plot_pareto_leaf_methods(ax[1, 3], demog)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "condition_summary")


if __name__ == "__main__":
    main()

