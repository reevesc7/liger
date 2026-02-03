from typing import Any, Iterator, Sequence, overload
from dataclasses import dataclass
import argparse
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import liger.tpot.output_processing as op
from liger import plotting as pl
from liger import dataset as ds


@dataclass(slots=True)
class Config:
    tpot_dir: Path
    tpot_output_dir: Path
    output_dir: Path
    retrieve_runs: bool
    make_plots: bool
    run_id_filter: str
    run_data_filters: list[Any]
    summary_csv: Path
    summary_json: Path


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
    tpot_dir = Path(cfg["tpot_dir"])
    return Config(
        tpot_dir=tpot_dir,
        tpot_output_dir=tpot_dir / "outputs",
        output_dir=Path(cfg["output_dir"]),
        retrieve_runs=cfg["retrieve_runs"],
        make_plots=cfg["make_plots"],
        run_id_filter=cfg["run_id_filter"],
        run_data_filters=cfg["run_data_filters"],
        summary_csv=Path(cfg["output_dir"], cfg["summary_stem"]).with_suffix(".csv"),
        summary_json=Path(cfg["output_dir"], cfg["summary_stem"]).with_suffix(".json"),
    )


def mean_gen_time(manager_attributes: dict) -> float:
    return sum(manager_attributes["segment_run_times"]) / manager_attributes["complete_gens"]


@overload
def scores_mean(scores: dict[str, list[float]]) -> list[float]: ...
@overload
def scores_mean(scores: dict[str, None]) -> None: ...
def scores_mean(scores: dict[str, list[float]] | dict[str, None]) -> list[float] | None:
    return np.array([score for score in scores.values()]).mean(axis=0).tolist()


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
        match = re.search(r"\b([A-Z][A-Za-z0-9_]*)_1", full_str)
        if match is None:
            raise ValueError(error_msg)
        return match.group(1)
    if full_str[0] == "P":
        matches = re.findall(r"\b([A-Z][A-Za-z0-9_]*)", full_str)
        if len(matches) == 0:
            raise ValueError(error_msg)
        return _blacklist_search(matches, reverse=True)
    raise ValueError(error_msg)


def order_responses(folds: Sequence[dict[str, Any]]) -> list[Any]:
    responses: dict[str, Any] = {}
    for fold in folds:
        responses.update(fold)
    return [responses[key] for key in sorted(responses.keys(), key=lambda k: int(k))]


# TODO: remove unused function or alter functionality
def run_responses(
    id: str,
    kfold_predictions: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, list[Any]]:
    responses: dict[str, list[Any]] = {}
    for rand_state, folds in kfold_predictions.items():
        responses[f"{id}_{rand_state}"] = order_responses(folds)
    return responses


def retrieve_runs(cfg: Config) -> None:
    summary = {key: [] for key in (
        "id",
        "config_file",
        "data_file",
        "feature_keys",
        "target_keys",
        "feature_transformers",
        "target_transformers",
        "feature_transformers_kwargs",
        "target_transformers_kwargs",
        "random_state",
        "classification",
        "cv",
        "population_size",
        "max_time_mins",
        "max_eval_time_mins",
        "target_gens",
        "gens_per_segment",
        "early_stop",
        "complete_gens",
        "mean_gen_time",
        "scorers",
        "scorers_weights",
        "mean_score",
        "fitted_pipeline"
    )}
    data = op.filtered_runs(
        paths=cfg.tpot_output_dir,
        id_filter=cfg.run_id_filter + "/manager_data.json",
        data_filters=cfg.run_data_filters,
    )
    responses = {}
    for run_data in data:
        manager_parameters = run_data.get("manager_parameters")
        tpot_parameters = run_data.get("tpot_parameters")
        manager_attributes = run_data.get("manager_attributes")
        tpot_attributes = run_data.get("tpot_attributes")
        summary["id"].append(manager_parameters.get("id"))
        summary["config_file"].append(manager_parameters.get("config_file"))
        summary["data_file"].append(manager_parameters.get("data_file"))
        summary["feature_keys"].append(manager_parameters.get("feature_keys"))
        summary["target_keys"].append(manager_parameters.get("target_keys"))
        summary["feature_transformers"].append(manager_parameters.get("feature_transformers"))
        summary["target_transformers"].append(manager_parameters.get("target_transformers"))
        summary["feature_transformers_kwargs"].append(manager_parameters.get("feature_transformers_kwargs"))
        summary["target_transformers_kwargs"].append(manager_parameters.get("target_transformers_kwargs"))
        summary["random_state"].append(tpot_parameters.get("random_state"))
        summary["classification"].append(tpot_parameters.get("classification"))
        summary["cv"].append(tpot_parameters.get("cv"))
        summary["population_size"].append(tpot_parameters.get("population_size"))
        summary["max_time_mins"].append(tpot_parameters.get("max_time_mins"))
        summary["max_eval_time_mins"].append(tpot_parameters.get("max_eval_time_mins"))
        summary["target_gens"].append(manager_parameters.get("target_gens"))
        summary["gens_per_segment"].append(tpot_parameters.get("generations"))
        summary["early_stop"].append(tpot_parameters.get("early_stop"))
        summary["complete_gens"].append(manager_attributes.get("complete_gens"))
        summary["mean_gen_time"].append(mean_gen_time(manager_attributes))
        summary["scorers"].append(tpot_parameters.get("scorers"))
        summary["scorers_weights"].append(tpot_parameters.get("scorers_weights"))
        summary["mean_score"].append(scores_mean(manager_attributes.get("kfold_scores")))
        summary["fitted_pipeline"].append(find_root(tpot_attributes.get("fitted_pipeline_")))
        responses[manager_parameters.get("id")] = {
            "data_file": manager_parameters.get("data_file"),
            "target_keys": manager_parameters.get("target_keys"),
            "target_transformers": manager_parameters.get("target_transformers"),
            "target_transformers_kwargs": manager_parameters.get("target_transformers_kwargs"),
            "kfold_scores": manager_attributes.get("kfold_scores"),
            "kfold_predictions": manager_attributes.get("kfold_predictions"),
        }
        # kfold_predictions = manager_attributes.get("kfold_predictions")
        # if kfold_predictions:
        #     responses.update(run_responses(summary["id"][-1], kfold_predictions))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    # print(summary)
    summary.to_csv(cfg.summary_csv, index=False)
    # responses = pd.DataFrame(responses)
    # print(responses)
    # responses.to_csv(cfg.summary_json, index=False)
    with open(cfg.summary_json, "w") as f:
        json.dump(responses, f, indent=4)


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


def make_plots(cfg: Config):
    # TODO: Plot only errors per response, using the scorer(s) from training.
    dataset = ds.Dataset.from_csv(
        cfg.dataset_file,
        "no_match!@#",
        "logprob",
        y_transformers=[
            "liger.probabilities.apply_softmax",
            "liger.probabilities.apply_logprobs_mode",
            "liger.probabilities.apply_logprobs_mean",
            "liger.probabilities.apply_logprobs_variance",
            "liger.probabilities.apply_logprobs_std_dev",
        ],
        y_transformers_kwargs=[
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
            {"temperature": cfg.softmax_temperature},
        ]
    )
    responses = pd.read_csv(cfg.responses_file)
    means = pd.concat([pd.Series(dataset.y["mean"])] * responses.shape[1], axis=1)
    std_devs = pd.concat([pd.Series(dataset.y["std_dev"])] * responses.shape[1], axis=1)
    means.columns = responses.columns
    std_devs.columns = responses.columns
    responses_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "20_responses")
    abs_errors_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "30_abs_errors")
    squared_errors_fig(responses, means, cfg.dataset).savefig(cfg.output_dir / "31_squared_errors")
    zscores_fig(responses, means, std_devs, cfg.dataset).savefig(cfg.output_dir / "32_zscores")


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

