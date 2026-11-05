from typing import cast
from collections.abc import Hashable
import contextlib
from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cycler import cycler
from liger.pareto import pareto_knee
from liger import plotting as lgplot
from liger.tpot import data
from liger.tpot.data import (
    RunCol,
    IndivCol,
    NodeCol,
    ScoreValuation,
    GraphsBreakdown,
)


@dataclass(slots=True)
class Config:
    result_path: Path
    runs_indivs: dict[str, pd.DataFrame]
    score_0_neg: bool
    score_1_neg: bool
    score_0_log: bool
    score_1_log: bool

    def __init__(self, cfg_path: Path) -> None:
        self.result_path = cfg_path.with_suffix(".png")
        parsed = tomllib.load(cfg_path.open("rb"))
        self.runs_indivs = {}
        for dir in Path().glob(parsed["runs_dirs"]):
            print(f"loading {dir}")
            try:
                self.runs_indivs[dir.name] = data.load_indivs(dir)
            except FileNotFoundError:
                print("    missing data")
                pass
        self.score_0_neg = parsed.get("score_0_neg", False)
        self.score_1_neg = parsed.get("score_1_neg", False)
        self.score_0_log = parsed.get("score_0_log", False)
        self.score_1_log = parsed.get("score_1_log", False)


plt.rc("axes", prop_cycle=cycler(linestyle=["-", "--", "-."]) * cycler(color=[
    color["color"]
    for color in mpl.rcParams["axes.prop_cycle"]
]))


def _run_gen_ind_ns(
    data: pd.DataFrame | pd.Series,
    indivs: pd.DataFrame,
) -> pd.Series:
    return data.join(indivs[IndivCol.GEN]).groupby([
        RunCol.ID,
        IndivCol.GEN,
        IndivCol.ID,
    ]).size()


def _runs_xs(data: pd.DataFrame | pd.Series) -> dict[Hashable, pd.Index]:
    return data.groupby(RunCol.ID).groups


def _plot_run_gen_mean_sem(ax: Axes, data: pd.Series) -> None:
    run_gen_groups = data.groupby([
        RunCol.ID,
        IndivCol.GEN,
    ])
    run_gen_means = run_gen_groups.mean()
    run_gen_95ci = 1.96 * run_gen_groups.sem()
    for run_id, xs in _runs_xs(run_gen_means).items():
        run_means = run_gen_means.loc[xs]
        run_sems = run_gen_95ci.loc[xs]
        gens = run_means.index.get_level_values(IndivCol.GEN)
        ax.plot(gens, run_means, label=run_id)
        ax.fill_between(
            gens,
            run_means - run_sems,
            run_means + run_sems,
            alpha=0.2,
        )


def plot_n_nodes(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("number of nodes per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of nodes (mean, 95% CI)")
    run_gen_ind_n_nodes = _run_gen_ind_ns(graphs.nodes, graphs.indivs)
    _plot_run_gen_mean_sem(ax, run_gen_ind_n_nodes)


def plot_n_branches(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("number of branches per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("number of branches (mean, 95% CI)")
    run_gen_ind_n_branches = (_run_gen_ind_ns(graphs.edges, graphs.indivs)
        - _run_gen_ind_ns(graphs.nodes, graphs.indivs))
    _plot_run_gen_mean_sem(ax, run_gen_ind_n_branches)


def plot_eval_times(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("evaluation time per generation, by run")
    ax.set_xlabel("generation")
    ax.set_ylabel("evaluation time (s) (mean, 95% CI)")
    gen_indivs = graphs.indivs.set_index(IndivCol.GEN, append=True)
    run_gen_eval_times = gen_indivs[IndivCol.COMP_TIME] - gen_indivs[IndivCol.SUB_TIME]
    _plot_run_gen_mean_sem(ax, run_gen_eval_times)


def _set_scores_scales(cfg: Config, ax: Axes) -> Axes:
    if cfg.score_0_log:
        ax.set_xscale("log")
    if cfg.score_1_log:
        ax.set_yscale("log")
    return ax


def _negate_scores(cfg: Config, scores: pd.DataFrame) -> pd.DataFrame:
    if cfg.score_0_neg:
        scores.iloc[:, 0] = -1 * scores.iloc[:, 0]
    if cfg.score_1_neg and scores.shape[1] > 1:
        scores.iloc[:, 1] = -1 * scores.iloc[:, 1]
    return scores


def plot_pareto_scores(cfg: Config, ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("Pareto front scores, by run")
    ax = _set_scores_scales(cfg, ax)
    score_cols = data.score_columns(graphs.indivs)
    if len(score_cols) > 2:
        print("Scores are greater than 2 dimensions; skipping Pareto front scores plot")
        return
    pareto_mask = data.pareto_mask(graphs.indivs)
    scores = _negate_scores(cfg, graphs.indivs.loc[pareto_mask, score_cols])
    runs_xs = _runs_xs(scores)
    if scores.shape[1] == 1:
        ax.set_xlabel("run")
        ax.set_ylabel(str(scores.columns[0]))
        for index, (run_id, xs) in enumerate(runs_xs.items()):
            ax.bar(index, scores.loc[xs])
        return
    ax.set_xlabel(str(scores.columns[0]))
    ax.set_ylabel(str(scores.columns[1]))
    for run_id, xs in runs_xs.items():
        run_scores = scores.loc[xs]
        ax.scatter(run_scores.iloc[:, 0], run_scores.iloc[:, 1], alpha=0.2)
    # ax.set_xbound(-0.11, -0.08)
    # ax.set_ybound(-100, 15000)


def plot_pareto_nodes_branches(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("Pareto front number of nodes and branches, by run")
    ax.set_xlabel("number of nodes")
    ax.set_ylabel("number of branches")
    pareto_mask = data.pareto_mask(graphs.indivs)
    pareto_indivs = graphs.indivs[pareto_mask]
    pareto_nodes = graphs.nodes[pareto_mask.reindex(graphs.nodes.index)]
    pareto_edges = graphs.edges[pareto_mask.reindex(graphs.edges.index)]
    run_gen_ind_n_nodes = _run_gen_ind_ns(pareto_nodes, pareto_indivs)
    run_gen_ind_n_branches = (_run_gen_ind_ns(pareto_edges, pareto_indivs)
        - _run_gen_ind_ns(graphs.nodes, graphs.indivs))
    nodes_branches = pd.concat([
        run_gen_ind_n_nodes,
        run_gen_ind_n_branches,
    ], axis=1)
    for run_id, xs in _runs_xs(nodes_branches).items():
        run_individuals = nodes_branches.loc[xs]
        ax.scatter(run_individuals.iloc[:, 0], run_individuals.iloc[:, 1], alpha=0.2)


def _objectives_weights(graphs: GraphsBreakdown) -> list[int]:
    score_cols = data.score_columns(graphs.indivs)
    if graphs.score_valuation == ScoreValuation.GREATER:
        objectives_weights = [1 for _ in range(len(score_cols))]
        return objectives_weights
    elif graphs.score_valuation == ScoreValuation.LESSER:
        objectives_weights = [-1 for _ in range(len(score_cols))]
        return objectives_weights
    objectives_weights = []
    for score_col in score_cols:
        if all(graphs.indivs[score_col].dropna() <= 0.0):
            objectives_weights.append(1)
        elif all(graphs.indivs[score_col].dropna() >= 0.0):
            objectives_weights.append(-1)
        else:
            raise ValueError("Cannot determine valuation of objective; "
                f"{score_col!r} contains positive and negative values")
    return objectives_weights


def plot_pareto_knee_scores(
    cfg: Config,
    ax: Axes,
    graphs: GraphsBreakdown,
) -> None:
    score_cols = data.score_columns(graphs.indivs)
    if len(score_cols) > 2:
        print("Scores are greater than 2 dimensions; skipping Pareto knee scores plot")
        return
    if len(score_cols) == 1:
        plot_pareto_scores(cfg, ax, graphs)
        return
    scores = _negate_scores(
        cfg,
        graphs.indivs.loc[data.pareto_mask(graphs.indivs), score_cols],
    )
    ax.set_title("Pareto knee scores (1 per run)")
    ax.set_xlabel(str(scores.columns[0]) + " +/- 95% CI")
    ax.set_ylabel(str(scores.columns[1]) + " +/- 95% CI")
    ax = _set_scores_scales(cfg, ax)
    objectives_weights = _objectives_weights(graphs)
    knees_list: list[pd.Series] = []
    for run_id, xs in _runs_xs(scores).items():
        run_scores = scores.loc[xs]
        knees_list.append(pareto_knee(run_scores, objectives_weights))
    knees = pd.concat(knees_list, axis=1).T
    ax.scatter(knees.iloc[:, 0], knees.iloc[:, 1], alpha=0.5)
    xmin = knees.iloc[:, 0].min()
    xmax = knees.iloc[:, 0].max()
    xmean = knees.iloc[:, 0].mean()
    x95ci = 1.96 * cast(float, knees.iloc[:, 0].sem())
    ymin = knees.iloc[:, 1].min()
    ymax = knees.iloc[:, 1].max()
    ymean = knees.iloc[:, 1].mean()
    y95ci = 1.96 * cast(float, knees.iloc[:, 1].sem())
    ax.vlines(xmean, ymin, ymax)
    ax.axvspan(xmean - x95ci, xmean + x95ci, alpha=0.1)
    ax.hlines(ymean, xmin, xmax)
    ax.axhspan(ymean - y95ci, ymean + y95ci, alpha=0.1)


def _plot_methods_per_indiv(
    ax: Axes,
    nodes: pd.DataFrame,
    indivs: pd.DataFrame,
) -> None:
    n_indivs = indivs.groupby(IndivCol.GEN).size()
    p_methods = nodes.join(indivs[IndivCol.GEN]).groupby(
        [IndivCol.GEN, NodeCol.TYPE],
        sort=False,
    ).size().unstack(level=0, fill_value=0) / n_indivs
    p_methods = p_methods.loc[p_methods.sum(axis=1).sort_values(ascending=False).index]
    for method, p_method in p_methods.iterrows():
        ax.plot(p_method, label=method.__name__)


def plot_node_methods(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("nodes of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("nodes per individual (mean)")
    _plot_methods_per_indiv(ax, graphs.nodes, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def plot_root_methods(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("roots of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("roots per individual (mean)")
    _plot_methods_per_indiv(ax, graphs.roots, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def plot_leaf_methods(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("leaves of method per individual, by method")
    ax.set_xlabel("generation")
    ax.set_ylabel("leaves per individual (mean)")
    _plot_methods_per_indiv(ax, graphs.leaves, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def _plot_pareto_methods(
    ax: Axes,
    nodes: pd.DataFrame | pd.Series,
    indivs: pd.DataFrame | pd.Series,
) -> None:
    n_indivs = indivs.groupby(RunCol.ID).size()
    p_methods = nodes.groupby([RunCol.ID, NodeCol.TYPE], sort=False).size().unstack(
        level=0,
        fill_value=0,
    ) / n_indivs
    p_methods = p_methods.sort_index(axis=1)
    p_methods = p_methods.loc[p_methods.sum(axis=1).sort_values(ascending=False).index]
    x_ticks = [str(index) for index in range(p_methods.shape[1])]
    ax = lgplot.heat_map_axes(ax, x_ticks, p_methods.index.to_list())
    ax.imshow(p_methods)


def plot_pareto_root_methods(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("roots of method per Pareto individual, by method")
    ax.set_xlabel("run")
    ax.set_ylabel("roots per individual (mean)")
    pareto_mask = data.pareto_mask(graphs.indivs)
    _plot_pareto_methods(ax, graphs.roots[pareto_mask], graphs.indivs[pareto_mask])


def plot_pareto_leaf_methods(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("leaves of method per Pareto individual, by method")
    ax.set_xlabel("run")
    ax.set_ylabel("leaves per individual (mean)")
    pareto_mask = data.pareto_mask(graphs.indivs)
    _plot_pareto_methods(
        ax,
        graphs.leaves[pareto_mask.reindex(graphs.leaves.index)],
        graphs.indivs[pareto_mask],
    )


def make_figure(cfg: Config, graphs: GraphsBreakdown) -> Figure:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(40,20))
    fig.suptitle(cfg.result_path.stem)
    plot_n_nodes(ax[0, 0], graphs)
    plot_n_branches(ax[0, 1], graphs)
    plot_eval_times(ax[0, 1], graphs)
    plot_pareto_scores(cfg, ax[0, 2], graphs)
    # plot_pareto_nodes_branches(ax[0, 3], cfg.pareto_scale, graphs)
    plot_pareto_knee_scores(cfg, ax[0, 3], graphs)
    # plot_node_methods(ax[1, 0], runs_graphs)
    plot_root_methods(ax[1, 0], graphs)
    plot_leaf_methods(ax[1, 1], graphs)
    plot_pareto_root_methods(ax[1, 2], graphs)
    plot_pareto_leaf_methods(ax[1, 3], graphs)
    fig.tight_layout()
    return fig


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        graphs = GraphsBreakdown(cfg.runs_indivs)
        fig = make_figure(cfg, graphs)
        fig.savefig(cfg.result_path)


if __name__ == "__main__":
    main()
