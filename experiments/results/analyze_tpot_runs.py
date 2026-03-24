from typing import Any, Iterable, Iterator, Sequence
from dataclasses import dataclass
import argparse
from pathlib import Path
from statistics import mean
from random import random, randint
import json
import re
import ast
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.figure import Figure
import liger.tpot.output_processing as op
from liger import plotting as pl
from liger import dataset as ds
from liger.probabilities import softmax


@dataclass(slots=True)
class Config:
    output_dir: Path
    tpot_dir: Path
    runs_dir: Path
    retrieve_runs: bool
    make_plots: bool
    run_id_filter: str
    run_data_filters: list[Any]
    summary_csv: Path
    summary_json: Path
    softmax_temperature: float


@dataclass(slots=True)
class RunScores:
    mean_scores: dict[str, float]
    mean_samples_scores: dict[str, list[float]]


def init_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file path",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser) -> Path:
    args = parser.parse_args()
    return Path(args.config)


def read_config(config_file: str | Path) -> Config:
    with open(config_file, "r") as file:
        cfg = json.load(file)
    return Config(
        output_dir=Path(cfg["output_dir"]),
        tpot_dir=Path(cfg["tpot_dir"]),
        runs_dir=Path(cfg["runs_dir"]),
        retrieve_runs=cfg["retrieve_runs"],
        make_plots=cfg["make_plots"],
        run_id_filter=cfg["run_id_filter"],
        run_data_filters=cfg["run_data_filters"],
        summary_csv=Path(cfg["output_dir"], cfg["summary_stem"]).with_suffix(".csv"),
        summary_json=Path(cfg["output_dir"], cfg["summary_stem"]).with_suffix(".json"),
        softmax_temperature=cfg["softmax_temperature"],
    )


def _mean_gen_time(manager_attributes: dict) -> float:
    return sum(manager_attributes["segment_run_times"]) / manager_attributes["complete_gens"]


def _scores_mean(scores: dict[str, dict[str, list[float]]] | dict[str, None]) -> dict[str, float]:
    return {
        scorer: mean([
            mean(random_state)
            for random_state in scorer_scores.values()
        ])
        for scorer, scorer_scores in scores.items()
        if scorer_scores is not None
    }
    # return np.array([score for score in scores.values()]).mean(axis=0).tolist()


def _blacklist_search(items: Sequence[str], reverse: bool = False) -> str:
    blacklist = (
        "False",
        "True_",
        "C",
    )
    if reverse:
        it_var = -1
    else:
        it_var = 0
    for item in items[it_var::it_var]:
        if item not in blacklist:
            return item
    raise ValueError(f"No appropriate items found in {items}")


def find_root(fitted_pipeline: str | Sequence[str]) -> str:
    """Find the root `EstimatorNode` of a `GraphPipeline` or `Pipeline`, given a string representation.
    If the pipeline is a `GraphPipeline`, this only works if its root is an `EstimatorNode`.
    """
    full_str = "".join(fitted_pipeline)
    error_msg = f"Malformed fitted_pipeline string:\n    {full_str}"
    if full_str[0] == "[":
        graph = True
    elif full_str[0] == "P":
        graph = False
    else:
        raise ValueError(error_msg)
    if graph:
        match = re.search(r"\b([A-Z][A-Za-z0-9_]*)_1", full_str)
        if match is None:
            raise ValueError(error_msg)
        return match.group(1)
    else:
        matches = re.findall(r"\b([A-Z][A-Za-z0-9_]*)", full_str)
        if len(matches) == 0:
            raise ValueError(error_msg)
        return _blacklist_search(matches, reverse=True)


@dataclass(slots=True)
class PipelineSummary:
    root: list[dict[str, int | str]]
    nodes: list[dict[str, int | str]]
    connections: list[dict[str, int | str]]
    branches: list[dict[str, int | str]]


def wout_id(node: str) -> str:
    return node.rsplit("_", 1)[0]


def decomp_graph_pipeline(generation: int, pipeline: str | Sequence[str]) -> PipelineSummary:
    full_str = "".join(pipeline)
    pipe: list[tuple[str, str]] | list[str] = ast.literal_eval(full_str)
    if isinstance(pipe[0], str):
        all_nodes = {node for node in pipe}
        root = [{"generation": generation, "node": wout_id(pipe[0])}]
        nodes = [{"generation": generation, "node": wout_id(pipe[0])}]
        connections = [{}]
        branches = [{}]
    else:
        all_nodes = {node for connxn in pipe for node in connxn}
        root = [{"generation": generation, "node": wout_id(pipe[0][0])}]
        nodes = [
            {"generation": generation, "node": wout_id(node)}
            for node in all_nodes
        ]
        connections = [
            {"generation": generation, "receiver": wout_id(connxn[0]), "sender": wout_id(connxn[1])}
            for connxn in pipe
        ]
        receivers = [connxn[0] for connxn in pipe]
        branches: list[dict[str, int | str]] = []
        unique_receivers: set[str] = set()
        unique_branches: set[str] = set()
        for receiver in receivers:
            if receiver in unique_receivers and receiver not in unique_branches:
                branches.append({"generation": generation, "node": wout_id(receiver)})
                unique_branches.add(receiver)
                continue
            unique_receivers.add(receiver)
    return PipelineSummary(root, nodes, connections, branches)


def df_subgroup_proportions(df: pd.DataFrame, *groups: str) -> pd.DataFrame:
    subgroup_ns = (
        df
        .groupby([group for group in groups])
        .size()
        .rename("subgroup_n")
        .reset_index()
    )
    subgroup_ns["proportion"] = subgroup_ns["subgroup_n"] / subgroup_ns.groupby(groups[0])["subgroup_n"].transform("sum")
    return subgroup_ns


def retrieve_runs(cfg: Config) -> None:
    summary: list[dict[str, Any]] = []
    # summary = {key: [] for key in (
    #     "id",
    #     "config_file",
    #     "data_file",
    #     "feature_keys",
    #     "target_keys",
    #     "feature_transformers",
    #     "target_transformers",
    #     "feature_transformers_kwargs",
    #     "target_transformers_kwargs",
    #     "random_state",
    #     "classification",
    #     "cv",
    #     "population_size",
    #     "max_time_mins",
    #     "max_eval_time_mins",
    #     "target_gens",
    #     "gens_per_segment",
    #     "early_stop",
    #     "complete_gens",
    #     "mean_gen_time",
    #     "scorers",
    #     "scorers_weights",
    #     "mean_score",
    #     "fitted_pipeline"
    # )}
    data = op.filtered_runs(
        paths=cfg.runs_dir,
        id_filter=cfg.run_id_filter + "/manager_data.json",
        data_filters=cfg.run_data_filters,
    )
    summary_obj = {}
    pop_roots = []
    pop_nodes = []
    pop_connections = []
    pop_branches = []
    top_roots = []
    top_nodes = []
    top_connections = []
    top_branches = []
    for run_data in data:
        manager_parameters = run_data.get("manager_parameters")
        tpot_parameters = run_data.get("tpot_parameters")
        manager_attributes = run_data.get("manager_attributes")
        tpot_attributes = run_data.get("tpot_attributes")
        summary.append({
            "id": manager_parameters.get("id"),
            "config_file": manager_parameters.get("config_file"),
            "data_file": manager_parameters.get("data_file"),
            "feature_keys": manager_parameters.get("feature_keys"),
            "target_keys": manager_parameters.get("target_keys"),
            "feature_transformers": manager_parameters.get("feature_transformers"),
            "target_transformers": manager_parameters.get("target_transformers"),
            "feature_transformers_kwargs": manager_parameters.get("feature_transformers_kwargs"),
            "target_transformers_kwargs": manager_parameters.get("target_transformers_kwargs"),
            "random_state": tpot_parameters.get("random_state"),
            "classification": tpot_parameters.get("classification"),
            "cv": tpot_parameters.get("cv"),
            "population_size": tpot_parameters.get("population_size"),
            "max_time_mins": tpot_parameters.get("max_time_mins"),
            "max_eval_time_mins": tpot_parameters.get("max_eval_time_mins"),
            "target_gens": manager_parameters.get("target_gens"),
            "gens_per_segment": tpot_parameters.get("generations"),
            "early_stop": tpot_parameters.get("early_stop"),
            "complete_gens": manager_attributes.get("complete_gens"),
            "mean_gen_time": _mean_gen_time(manager_attributes),
            "scorers": tpot_parameters.get("scorers"),
            "scorers_weights": tpot_parameters.get("scorers_weights"),
            "mean_score": _scores_mean(manager_attributes.get("kfold_scores", {})),
            "fitted_pipeline": find_root(tpot_attributes.get("fitted_pipeline_")),
        })
        # summary["id"].append(manager_parameters.get("id"))
        # summary["config_file"].append(manager_parameters.get("config_file"))
        # summary["data_file"].append(manager_parameters.get("data_file"))
        # summary["feature_keys"].append(manager_parameters.get("feature_keys"))
        # summary["target_keys"].append(manager_parameters.get("target_keys"))
        # summary["feature_transformers"].append(manager_parameters.get("feature_transformers"))
        # summary["target_transformers"].append(manager_parameters.get("target_transformers"))
        # summary["feature_transformers_kwargs"].append(manager_parameters.get("feature_transformers_kwargs"))
        # summary["target_transformers_kwargs"].append(manager_parameters.get("target_transformers_kwargs"))
        # summary["random_state"].append(tpot_parameters.get("random_state"))
        # summary["classification"].append(tpot_parameters.get("classification"))
        # summary["cv"].append(tpot_parameters.get("cv"))
        # summary["population_size"].append(tpot_parameters.get("population_size"))
        # summary["max_time_mins"].append(tpot_parameters.get("max_time_mins"))
        # summary["max_eval_time_mins"].append(tpot_parameters.get("max_eval_time_mins"))
        # summary["target_gens"].append(manager_parameters.get("target_gens"))
        # summary["gens_per_segment"].append(tpot_parameters.get("generations"))
        # summary["early_stop"].append(tpot_parameters.get("early_stop"))
        # summary["complete_gens"].append(manager_attributes.get("complete_gens"))
        # summary["mean_gen_time"].append(mean_gen_time(manager_attributes))
        # summary["scorers"].append(tpot_parameters.get("scorers"))
        # summary["scorers_weights"].append(tpot_parameters.get("scorers_weights"))
        # summary["mean_score"].append(scores_mean(manager_attributes.get("kfold_scores")))
        # summary["fitted_pipeline"].append(find_root(tpot_attributes.get("fitted_pipeline_")))
        summary_obj[manager_parameters.get("id")] = {
            "data_file": manager_parameters.get("data_file"),
            "target_keys": manager_parameters.get("target_keys"),
            "target_transformers": manager_parameters.get("target_transformers"),
            "target_transformers_kwargs": manager_parameters.get("target_transformers_kwargs"),
            "kfold_scores": manager_attributes.get("kfold_scores"),
            "kfold_samples_scores": manager_attributes.get("kfold_samples_scores"),
            "kfold_predictions": manager_attributes.get("kfold_predictions"),
        }
        indivs = tpot_attributes["evaluated_individuals"]
        roots = []
        nodes = []
        connections = []
        branches = []
        for gen, graph in zip(indivs["Generation"].values(), indivs["Instance"].values()):
            graph_sum = decomp_graph_pipeline(int(gen), graph)
            roots.extend(graph_sum.root)
            nodes.extend(graph_sum.nodes)
            connections.extend(graph_sum.connections)
            branches.extend(graph_sum.branches)
        pop_roots.append(pd.DataFrame(roots))
        pop_nodes.append(pd.DataFrame(nodes))
        pop_connections.append(pd.DataFrame(connections))
        pop_branches.append(pd.DataFrame(branches))
        top_graph = decomp_graph_pipeline(0, tpot_attributes["fitted_pipeline_"])
        top_roots.append(pd.DataFrame(top_graph.root))
        top_nodes.append(pd.DataFrame(top_graph.nodes))
        top_connections.append(pd.DataFrame(top_graph.connections))
        top_branches.append(pd.DataFrame(top_graph.branches))
        # kfold_predictions = manager_attributes.get("kfold_predictions")
        # if kfold_predictions:
        #     responses.update(run_responses(summary["id"][-1], kfold_predictions))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pop_roots = pd.concat(pop_roots)
    pop_nodes = pd.concat(pop_nodes)
    pop_connections = pd.concat(pop_connections)
    pop_branches = pd.concat(pop_branches)
    pop_root_ns = df_subgroup_proportions(pop_roots, "generation", "node")
    pop_node_ns = df_subgroup_proportions(pop_nodes, "generation", "node")
    pop_connection_ns = df_subgroup_proportions(pop_connections, "generation", "receiver", "sender")
    pop_branch_ns = df_subgroup_proportions(pop_branches, "generation", "node")
    pop_root_ns.to_csv(cfg.output_dir / "pop_roots.csv", index=False)
    pop_node_ns.to_csv(cfg.output_dir / "pop_nodes.csv", index=False)
    pop_connection_ns.to_csv(cfg.output_dir / "pop_connections.csv", index=False)
    pop_branch_ns.to_csv(cfg.output_dir / "pop_branches.csv", index=False)
    top_roots = pd.concat(top_roots)
    top_nodes = pd.concat(top_nodes)
    top_connections = pd.concat(top_connections)
    top_branches = pd.concat(top_branches)
    top_root_ns = df_subgroup_proportions(top_roots, "generation", "node")
    top_node_ns = df_subgroup_proportions(top_nodes, "generation", "node")
    top_connection_ns = df_subgroup_proportions(top_connections, "generation", "receiver", "sender")
    top_branch_ns = df_subgroup_proportions(top_branches, "generation", "node")
    top_root_ns.to_csv(cfg.output_dir / "top_roots.csv", index=False)
    top_node_ns.to_csv(cfg.output_dir / "top_nodes.csv", index=False)
    top_connection_ns.to_csv(cfg.output_dir / "top_connections.csv", index=False)
    top_branch_ns.to_csv(cfg.output_dir / "top_branches.csv", index=False)
    # summary = pd.DataFrame(summary)
    # print(summary)
    pd.DataFrame(summary).to_csv(cfg.summary_csv, index=False)
    # responses = pd.DataFrame(responses)
    # print(responses)
    # responses.to_csv(cfg.summary_json, index=False)
    with open(cfg.summary_json, "w") as f:
        json.dump(summary_obj, f, indent=4)


def _order_samples_scores(folds: Sequence[dict[str, float | list[float]]]) -> list[float | list[float]]:
    # a single scorer
    # random state dict, list of folds, dict of scores by orignal index
    #
    # need to flatten list of folds... into list of scores
    # then average over random states
    #
    # one function to order them
    # one function to average them
    # one function to rule them all
    responses: dict[str, float | list[float]] = {}
    for fold in folds:
        responses.update(fold)
    return [responses[key] for key in sorted(responses.keys(), key=lambda k: int(k))]


def _collapse_folds(samples_scores: list[dict[str, float]]) -> list[float]:
    flat_scores: dict[str, float] = {}
    for fold in samples_scores:
        flat_scores.update(fold)
    return [flat_scores[key] for key in sorted(flat_scores.keys(), key=lambda k: int(k))]


def _scorer_samples_mean(samples_scores: dict[str, list[dict[str, float]]]) -> list[float]:
    collapsed_scores = [
        _collapse_folds(random_state_scores)
        for random_state_scores in samples_scores.values()
    ]
    return [
        mean(
            collapsed_scores[random_state][sample]
            for random_state in range(len(collapsed_scores))
        )
        for sample in range(len(collapsed_scores[0]))
    ]


def _samples_scores_mean(
        samples_scores: dict[str, dict[str, list[dict[str, float]]]],
) -> dict[str, list[float]]:
    return {
        scorer: _scorer_samples_mean(scorer_scores)
        for scorer, scorer_scores in samples_scores.items()
    }


def _get_run_scores(run_data: dict[str, Any]) -> RunScores | None:
    # TODO: maybe make these dicts with the scorer labeled, but also screw all this _Scorer bs.
    # TODO: functionalize this by caring about dataset, target_keys, and target_transformers.
    mean_scores = _scores_mean(run_data["kfold_scores"])
    if len(mean_scores) == 0:
        return None
    # responses = _get_responses(run_data["kfold_predictions"])
    # responses_scores = _score_responses(responses)
    mean_samples_scores = _samples_scores_mean(run_data["kfold_samples_scores"])
    # mean_response_scores = np.asarray(responses_scores).mean(axis=0).tolist()
    return RunScores(mean_scores, mean_samples_scores)


def _get_run_predictions(run_data: dict[str, Any]) -> dict[int, list[Any]] | None:
    return {
        int(kfold_rs): _collapse_folds(predictions)
        for kfold_rs, predictions in run_data["kfold_predictions"].items()
    }


def _load_dataset(cfg: Config, dataset_file: str | Path) -> ds.Dataset:
    return ds.Dataset.from_csv(
        cfg.tpot_dir / dataset_file,
        None,
        "logprob",
        y_transformers=[
            "liger.probabilities.apply_passthrough",
            "liger.probabilities.apply_logprobs_mode",
            "liger.probabilities.apply_logprobs_mean",
        ],
        y_transformers_kwargs=[
            {},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
        ]
    )


def responses_fig(
    responses: pd.DataFrame,
    means: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot predicted means against training means.
    """
    return pl.scatter(
        x=np.transpose(means),
        y=np.transpose(responses),
        title=f"{dataset}: predicted means against training means",
        axis_labels=("ChatGPT mean", "predicted"),
        trend_orders=[5],
        plot_perfect=True,
    )


def abs_errors_fig(
    responses: pd.DataFrame,
    means: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot absolute errors of predictions across training means.
    """
    abs_errors = np.absolute(responses - means)
    return pl.scatter(
        x=np.transpose(means),
        y=np.transpose(abs_errors),
        title=f"{dataset}: absolute errors of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted absolute error"),
        trend_orders=[5],
    )


def squared_errors_fig(
    responses: pd.DataFrame,
    means: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot squared errors of predictions across training means.
    """
    squared_errors = np.square(responses - means)
    return pl.scatter(
        x=np.transpose(means),
        y=np.transpose(squared_errors),
        title=f"{dataset}: squared errors of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted squared error"),
        trend_orders=[5],
    )


def zscores_fig(
    responses: pd.DataFrame,
    means: pd.DataFrame,
    std_devs: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot Z-scores of predictions across training means.
    """
    zscores = np.clip((responses - means) / std_devs, -20, 20)
    return pl.scatter(
        x=np.transpose(means),
        y=np.transpose(zscores),
        title=f"{dataset}: Z-scores of predicted responses against training means",
        axis_labels=("ChatGPT mean", "predicted Z-score"),
        trend_orders=[5],
    )


def sample_pmf_fig(
    target: ArrayLike,
    predicted: ArrayLike,
    temperature: float,
    score: float | None = None,
) -> Figure:
    target = np.asarray(target)
    predicted = np.asarray(predicted)
    support = np.arange(1, target.size + 1)
    target_pmf = np.stack((support, softmax(target, temperature)))
    predicted_pmf = np.stack((support + 0.5, softmax(predicted, temperature)))
    return pl.bar(
        data=[target_pmf, predicted_pmf],
        title=f"target-predicted PMF comparison; sample score: {score}",
        axis_labels=("support", "probability mass")
    )


def mean_scores_fig(
    data: ArrayLike,
    support: ArrayLike,
) -> Figure:
    data = np.asarray(data)
    support = np.asarray(support)
    mean = data.mean()
    sem = data.std() / len(support) ** 0.5
    return pl.bar(
        data=[np.stack((support, data))],
        title=f"Wasserstein score per run; mean, SEM run score: {mean}, {sem}",
        axis_labels=("run", "mean normalized Wasserstein distance"),
    )


def samples_scores_fig(
    data: Iterable[ArrayLike],
    scorer: str,
) -> Figure:
    return pl.scatter(
        data=data,
        title=f"{scorer}: samples scores",
        axis_labels=("ChatGPT mean", "score"),
        trend_orders=[5],
    )


def make_plots(cfg: Config):
    # TODO: Plot only errors per response and run errors.
    with open(cfg.summary_json, "r") as f:
        runs_data = json.load(f)
    assert isinstance(runs_data, dict), "ERROR: runs_data is not a dict"
    datasets = {}
    samples_means_plot_data = []
    mean_scores = []
    n_runs = 0
    n_plotted = 0
    for id, run_data in runs_data.items():
        dataset: ds.Dataset = datasets.setdefault(
            run_data["data_file"],
            _load_dataset(cfg, run_data["data_file"]),
        )
        run_scores = _get_run_scores(run_data)
        if run_scores is not None:
            n_runs += 1
            samples_means_plot_data.append([
                dataset.y["mean"],
                run_scores.mean_samples_scores["liger.objectives.neg_softmax_norm_wasserstein"],
            ])
            mean_scores.append(run_scores.mean_scores["liger.objectives.neg_softmax_norm_wasserstein"])
        run_predictions = _get_run_predictions(run_data)
        if run_predictions is not None:
            # if random() < 0.03:
            if n_plotted < 2:
                n_plotted += 1
                fold_1_preds = tuple(run_predictions.values())[0]
                index = randint(0, len(fold_1_preds))
                if run_scores is not None:
                    score = run_scores.mean_samples_scores["liger.objectives.neg_softmax_norm_wasserstein"][index]
                else:
                    score = None
                sample_pmf_fig(
                    dataset.y.iloc[index].filter(like="prob"),
                    fold_1_preds[index],
                    cfg.softmax_temperature,
                    score,
                ).savefig(cfg.output_dir / f"sample_pmf_{id}_{index}.png")
    samples_scores_fig(
        samples_means_plot_data,
        "liger.objectives.neg_softmax_norm_wasserstein",
    ).savefig(cfg.output_dir / "mean_samples_scores")
    mean_scores_fig(
        mean_scores,
        range(n_runs),
    ).savefig(cfg.output_dir / "mean_scores")
    # responses = pd.read_csv(cfg.responses_file)
    # means = pd.concat([pd.Series(dataset.y["mean"])] * responses.shape[1], axis=1)
    # std_devs = pd.concat([pd.Series(dataset.y["std_dev"])] * responses.shape[1], axis=1)
    # means.columns = responses.columns
    # std_devs.columns = responses.columns
    # responses_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "20_responses")
    # abs_errors_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "30_abs_errors")
    # squared_errors_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "31_squared_errors")
    # zscores_fig(responses, means, std_devs, cfg.dataset).savefig(cfg.output_dir / "32_zscores")


def main():
    arparser = init_argparser()
    cfg_file = parse_args(arparser)
    cfg = read_config(cfg_file)
    if cfg.retrieve_runs:
        retrieve_runs(cfg)
    if cfg.make_plots:
        make_plots(cfg)


if __name__ == "__main__":
    main()

