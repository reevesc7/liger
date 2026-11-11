from typing import cast
from collections.abc import Hashable
import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tomllib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from cycler import cycler
from liger import config
from liger import pareto
from liger import plotting as lgplot
from liger.tpot import data
from liger.tpot.data import (
    RunCol,
    IndivCol,
    NodeCol,
    EdgeCol,
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
    pareto_mask = graphs.indivs[IndivCol.PARETO] == 1
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
    pareto_mask = graphs.indivs[IndivCol.PARETO] == 1
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
    scores = graphs.indivs.loc[graphs.indivs[IndivCol.PARETO] == 1, score_cols]
    ax.set_title("Pareto knee scores (1 per run)")
    ax.set_xlabel(str(scores.columns[0]) + " +/- 95% CI")
    ax.set_ylabel(str(scores.columns[1]) + " +/- 95% CI")
    ax = _set_scores_scales(cfg, ax)
    maximizing_scores = scores * _objectives_weights(graphs)
    knees_indices = []
    for run_id, xs in _runs_xs(scores).items():
        run_scores = maximizing_scores.loc[xs]
        knee = pareto.pareto_knee(run_scores)
        if knee.status == pareto.KneeStatus.DEGENERATE:
            raise ValueError(f"Found degenerate Pareto extremes in {run_id}")
        print(f"{run_id}: [ n_knees: {knee.indices.size}, status: {knee.status} ]")
        for index in knee.indices:
            knees_indices.append(run_scores.index[index])
    knees_scores = _negate_scores(cfg, scores.loc[knees_indices])
    ax.scatter(knees_scores.iloc[:, 0], knees_scores.iloc[:, 1], alpha=0.5)
    xmin = knees_scores.iloc[:, 0].min()
    xmax = knees_scores.iloc[:, 0].max()
    xmean = knees_scores.iloc[:, 0].mean()
    x95ci = 1.96 * cast(float, knees_scores.iloc[:, 0].sem())
    ymin = knees_scores.iloc[:, 1].min()
    ymax = knees_scores.iloc[:, 1].max()
    ymean = knees_scores.iloc[:, 1].mean()
    y95ci = 1.96 * cast(float, knees_scores.iloc[:, 1].sem())
    ax.vlines(xmean, ymin, ymax)
    ax.axvspan(xmean - x95ci, xmean + x95ci, alpha=0.1)
    ax.hlines(ymean, xmin, xmax)
    ax.axhspan(ymean - y95ci, ymean + y95ci, alpha=0.1)


def _plot_types_per_indiv(
    ax: Axes,
    nodes: pd.DataFrame,
    indivs: pd.DataFrame,
) -> None:
    n_indivs = indivs.groupby(IndivCol.GEN).size()
    p_types = nodes.join(indivs[IndivCol.GEN]).groupby(
        [IndivCol.GEN, NodeCol.TYPE],
        sort=False,
    ).size().unstack(level=0, fill_value=0) / n_indivs
    p_types = p_types.loc[p_types.sum(axis=1).sort_values(ascending=False).index]
    for type_, p_type in p_types.iterrows():
        ax.plot(p_type, label=type_)


def plot_node_types(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("nodes of type per individual, by type")
    ax.set_xlabel("generation")
    ax.set_ylabel("nodes per individual (mean)")
    _plot_types_per_indiv(ax, graphs.nodes, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def plot_root_types(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("roots of type per individual, by type")
    ax.set_xlabel("generation")
    ax.set_ylabel("roots per individual (mean)")
    _plot_types_per_indiv(ax, graphs.roots, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def plot_leaf_types(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("leaves of type per individual, by type")
    ax.set_xlabel("generation")
    ax.set_ylabel("leaves per individual (mean)")
    _plot_types_per_indiv(ax, graphs.leaves, graphs.indivs)
    ax.legend(loc="upper left", framealpha=0)


def _plot_pareto_types(
    ax: Axes,
    nodes: pd.DataFrame | pd.Series,
    indivs: pd.DataFrame | pd.Series,
) -> None:
    n_indivs = indivs.groupby(RunCol.ID).size()
    p_types = nodes.groupby([RunCol.ID, NodeCol.TYPE], sort=False).size().unstack(
        level=0,
        fill_value=0,
    ) / n_indivs
    p_types = p_types.sort_index(axis=1)
    p_types = p_types.loc[p_types.sum(axis=1).sort_values(ascending=False).index]
    x_ticks = [str(index) for index in range(p_types.shape[1])]
    ax = lgplot.heat_map_axes(ax, x_ticks, p_types.index.to_list())
    ax.imshow(p_types)


def plot_pareto_root_types(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("roots of type per Pareto individual, by type")
    ax.set_xlabel("run")
    ax.set_ylabel("roots per individual (mean)")
    pareto_mask = graphs.indivs[IndivCol.PARETO] == 1
    _plot_pareto_types(ax, graphs.roots[pareto_mask], graphs.indivs[pareto_mask])


def plot_pareto_leaf_types(ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("leaves of type per Pareto individual, by type")
    ax.set_xlabel("run")
    ax.set_ylabel("leaves per individual (mean)")
    pareto_mask = graphs.indivs[IndivCol.PARETO] == 1
    _plot_pareto_types(
        ax,
        graphs.leaves[pareto_mask.reindex(graphs.leaves.index)],
        graphs.indivs[pareto_mask],
    )


def plot_pareto_graphs(cfg: Config, ax: Axes, graphs: GraphsBreakdown) -> None:
    ax.set_title("Pareto individual graph layout proportions, by run")
    ax.set_xlabel("run")
    ax.set_ylabel("graph layout proportion")
    score_cols = data.score_columns(graphs.indivs)
    scores = graphs.indivs.loc[graphs.indivs[IndivCol.PARETO] == 1, score_cols]
    maximizing_scores = scores * _objectives_weights(graphs)
    knees_indices = []
    for run_id, xs in _runs_xs(scores).items():
        run_scores = maximizing_scores.loc[xs]
        knee = pareto.pareto_knee(run_scores)
        if knee.status == pareto.KneeStatus.DEGENERATE:
            raise ValueError(f"Found degenerate Pareto extremes in {run_id}")
        for index in knee.indices:
            knees_indices.append(run_scores.index[index])
    knees = graphs.indivs.loc[knees_indices]
    knees[NodeCol.TYPE] = knees[IndivCol.INSTANCE].map(str)
    # TODO: Clean this up, move, refactor, etc.
    knee_store = {}
    for knee_index in knees_indices:
        edges = graphs.edges.xs(knee_index, level=[RunCol.ID, IndivCol.ID])
        edge_list = [
            f"({node}, {source[EdgeCol.SOURCE_ID]})"
            for node, source in edges.iterrows()
        ]
        nodes = graphs.nodes.xs(knee_index, level=[RunCol.ID, IndivCol.ID])
        node_list = nodes[NodeCol.TYPE].to_list()
        knee_store[str(knee_index)] = ({"nodes": node_list, "edges": edge_list})
    json.dump(knee_store, (cfg.result_path.parent / "test.json").open("w"), indent=4)
    return
    _plot_pareto_types(
        ax,
        knees,
        knees,
        # pareto_indivs,
        # pareto_indivs,
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
    # plot_node_types(ax[1, 0], runs_graphs)
    plot_root_types(ax[1, 0], graphs)
    plot_leaf_types(ax[1, 1], graphs)
    plot_pareto_root_types(ax[1, 2], graphs)
    plot_pareto_leaf_types(ax[1, 3], graphs)
    plot_pareto_graphs(cfg, ax[1, 3], graphs)
    fig.tight_layout()
    return fig


def report_total_knee(cfg: Config, graphs: GraphsBreakdown) -> None:
    pareto_mask = graphs.indivs[IndivCol.PARETO] == 1
    score_cols = data.score_columns(graphs.indivs)
    scores = graphs.indivs.loc[pareto_mask, score_cols]
    knee = pareto.pareto_knee(scores * _objectives_weights(graphs))
    if knee.status == pareto.KneeStatus.DEGENERATE:
        raise ValueError("Found degenerate Pareto extremes")
    print(f"Found {knee.indices.size} potential knee(s); status: {knee.status}")
    knee_indivs = graphs.indivs[pareto_mask].iloc[knee.indices]
    cfg.result_path.with_suffix(".json").write_text(
        json.dumps(config.to_config(knee_indivs), indent=4)
    )


def main():
    cfg_path = Path(sys.argv[1])
    with contextlib.chdir(cfg_path.parent):
        cfg = Config(Path(cfg_path.name))
        graphs = GraphsBreakdown(cfg.runs_indivs)
        fig = make_figure(cfg, graphs)
        fig.savefig(cfg.result_path)
        report_total_knee(cfg, graphs)


if __name__ == "__main__":
    main()
