from dataclasses import dataclass
from pathlib import Path
import json
import re
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from matplotlib.figure import Figure
import liger.output_processing as op
from liger import plotting as pl
from liger import dataset as ds


@dataclass(slots=True)
class Config:
    retrieve_runs: bool
    make_plots: bool
    dataset: str
    results_dir: Path
    dataset_file: Path
    outputs_dirs: list[Path]
    run_id_pattern: str
    summary_file: Path
    responses_file: Path


def read_config(config_file: str | Path) -> Config:
    with open(config_file, "r") as file:
        cfg = json.load(file)
    return Config(
        retrieve_runs=cfg["retrieve_runs"],
        make_plots=cfg["make_plots"],
        dataset=cfg["dataset"],
        results_dir=Path(cfg["dataset"]),
        dataset_file=Path(cfg["dataset_loc"], cfg["dataset"]).with_suffix(".csv"),
        outputs_dirs=[Path(path, cfg["dataset"]) for path in cfg["outputs_locs"]],
        run_id_pattern=cfg["run_id_pattern"],
        summary_file=Path(cfg["dataset"], cfg["summary"]).with_suffix(".csv"),
        responses_file=Path(cfg["dataset"], cfg["responses"]).with_suffix(".csv"),
    )


def find_root(fitted_pipeline: str | list[str]) -> str:
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
        return matches[-1]
    raise ValueError(error_msg)


def order_responses(folds: list[dict[str, float]]) -> list[float]:
    responses: dict[str, float] = {}
    for fold in folds:
        responses.update(fold)
    return [responses[key] for key in sorted(responses.keys(), key=lambda k: int(k))]


def run_responses(
    id: str,
    kfold_predictions: dict[str, list[dict[str, float]]],
) -> dict[str, list[float]]:
    responses: dict[str, list[float]] = {}
    for rand_state, folds in kfold_predictions.items():
        responses[f"{id}_{rand_state}"] = order_responses(folds)
    return responses


def retrieve_runs(cfg: Config):
    data = op.mass_json_load(paths=cfg.outputs_dirs, pattern=cfg.run_id_pattern)
    summary = {key: [] for key in ("id", "complete_gens", "score", "fitted_pipeline")}
    responses = {}
    for run_data in data:
        manager_parameters = run_data.get("pipeline_parameters")
        manager_attributes = run_data.get("pipeline_attributes")
        tpot_attributes = run_data.get("tpot_attributes")
        summary["id"].append(manager_parameters.get("id"))
        summary["complete_gens"].append(manager_attributes.get("complete_gens"))
        summary["score"].append(manager_attributes.get("gen_scores")[-1])
        summary["fitted_pipeline"].append(find_root(tpot_attributes.get("fitted_pipeline_")))
        kfold_predictions = manager_attributes.get("kfold_predictions")
        if kfold_predictions:
            responses.update(run_responses(summary["id"][-1], kfold_predictions))
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    print(summary)
    summary.to_csv(cfg.summary_file, index=False)
    responses = pd.DataFrame(responses)
    print(responses)
    responses.to_csv(cfg.responses_file, index=False)


def training_variances_fig(
    means: ArrayLike,
    std_devs: ArrayLike,
    dataset: str,
) -> Figure:
    """Plot the standard deviation of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=std_devs,
        title=f"{dataset}: ChatGPT responses, standard deviation by mean",
        axis_labels=("mean", "std_dev"),
        trend_orders=[],
        plot_perfect=False,
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


def make_plots(cfg: Config):
    dataset = ds.Dataset.from_csv(cfg.dataset_file, "no_match!@#", ["mean", "std_dev"])
    responses = pd.read_csv(cfg.responses_file)
    means = pd.concat([pd.Series(dataset.y["mean"])] * responses.shape[1], axis=1)
    std_devs = pd.concat([pd.Series(dataset.y["std_dev"])] * responses.shape[1], axis=1)
    means.columns = responses.columns
    std_devs.columns = responses.columns
    training_variances_fig(
        dataset.y["mean"],
        dataset.y["std_dev"],
        cfg.dataset,
    ).savefig(cfg.results_dir / "1_training_variances")
    responses_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "2_responses")
    abs_errors_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "3_abs_errors")
    squared_errors_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "4_squared_errors")
    zscores_fig(responses, means, std_devs, cfg.dataset).savefig(cfg.results_dir / "5_zscores")


def main():
    cfg = read_config("config.json")
    if cfg.retrieve_runs:
        retrieve_runs(cfg)
    if cfg.make_plots:
        make_plots(cfg)


if __name__ == "__main__":
    main()

