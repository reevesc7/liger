from typing import Iterable, Sequence
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


def mean_gen_time(manager_attributes: dict) -> float:
    return sum(manager_attributes["segment_run_times"]) / manager_attributes["complete_gens"]


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


def order_responses(folds: Sequence[dict[str, float]]) -> list[float]:
    responses: dict[str, float] = {}
    for fold in folds:
        responses.update(fold)
    return [responses[key] for key in sorted(responses.keys(), key=lambda k: int(k))]


def run_responses(
    id: str,
    kfold_predictions: dict[str, Sequence[dict[str, float]]],
) -> dict[str, list[float]]:
    responses: dict[str, list[float]] = {}
    for rand_state, folds in kfold_predictions.items():
        responses[f"{id}_{rand_state}"] = order_responses(folds)
    return responses


def retrieve_runs(cfg: Config) -> None:
    data = op.mass_json_load(paths=cfg.outputs_dirs, pattern=cfg.run_id_pattern)
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
        "score",
        "fitted_pipeline"
    )}
    responses = {}
    for run_data in data:
        manager_parameters = run_data.get("manager_parameters")
        tpot_parameters = run_data.get("tpot_parameters")
        manager_attributes = run_data.get("manager_attributes")

        # TODO: remove deprecation check in v0.9.0+
        if manager_parameters is None:
            manager_parameters = run_data.get("pipeline_parameters")
            if manager_parameters is not None:
                print("WARNING: using \"pipeline_parameters\" is deprecated. Use \"manager_parameters\" instead.", flush=True)
        if manager_attributes is None:
            manager_attributes = run_data.get("pipeline_attributes")
            if manager_attributes is not None:
                print("WARNING: using \"pipeline_attributes\" is deprecated. Use \"manager_attributes\" instead.", flush=True)

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
    """Plot the variances of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=np.asarray(std_devs) ** 2,
        title=f"{dataset}: ChatGPT responses, variance by mean",
        axis_labels=("mean", "variance"),
        trend_orders=[],
        plot_perfect=False,
    )


def _sq_mode_dist(row: pd.Series) -> float:
    return sum(
        float(row[f"prob_{i}"]) * (i - float(row["mode"])) ** 2
        for i in range(1,11)
    )


def training_sq_mode_dists_fig(
    means: ArrayLike,
    modes: pd.Series,
    probs: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot the expected squared distances to the mode of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=pd.concat((modes, probs), axis=1).apply(_sq_mode_dist, axis=1),
        title=f"{dataset}: ChatGPT responses, expected squared distance to the mode by mean",
        axis_labels=("mean", "sq_mode_dist"),
        trend_orders=[],
        plot_perfect=False,
    )


def training_mode_fig(
    means: ArrayLike,
    modes: ArrayLike,
    dataset: str,
) -> Figure:
    """Plot the modes of LLM response distributions, across means.
    """
    return pl.scatter(
        x=means,
        y=modes,
        title=f"{dataset}: ChatGPT responses, mode by mean",
        axis_labels=("mean", "mode"),
        trend_orders=[],
        plot_perfect=False,
    )


# def training_confidence_fig(
#     means: ArrayLike,
#     std_devs: ArrayLike,
#     dataset: str,
# ) -> Figure:
#     """Plot the confidences of LLM response distributions, across means.
#     """
#     return pl.scatter(
#         x=means,
#         y=np.full(std_devs.shape, 1) - np.asarray(std_devs) ** 2 / 20.25,
#         title=f"{dataset}: ChatGPT responses, confidence by mean",
#         axis_labels=("mean", "confidence"),
#         trend_orders=[],
#         plot_perfect=False,
#     )


def _agreement(row: pd.Series, target: str, interval: int = 1) -> float:
    response = int(row[target])
    return sum(float(row[f"prob_{i}"]) for i in range(
        max(response - interval, 1),
        min(response + interval + 1, 11),
    ))


def training_mean_agreement_fig(
    means: pd.Series,
    probs: pd.DataFrame,
    dataset: str,
) -> Figure:
    """Plot the agreements of LLM response distributions, across means.
    """
    targets = means.apply(lambda row: round(row))
    return pl.scatter(
        x=means,
        y=pd.concat((targets, probs), axis=1).apply(lambda row: _agreement(row, "mean"), axis=1),
        title=f"{dataset}: ChatGPT responses, agreement with mean by mean",
        axis_labels=("mean", "agreement"),
        trend_orders=[],
        plot_perfect=False,
    )


def _sem(row: pd.Series) -> pd.Series:
    return pd.Series(
        (row["mean"], row["std_dev"] / row["n"] ** 0.5),
        index=["mean", "sem"],
    )


def _mean_sem_of_2_grouped_by_1(data: pd.DataFrame, groups: Iterable) -> pd.DataFrame:
    return pd.DataFrame(data.groupby(data.columns[0]).agg(
        mean=(data.columns[1], "mean"),
        std_dev=(data.columns[1], "std"),
        n=(data.columns[1], "count"),
    ).apply(_sem, axis=1).reindex(groups).fillna(0.0))


def training_variances_mode_fig(
    data: pd.DataFrame,
    dataset: str,
) -> Figure:
    modes = range(1,11)
    data["std_dev"] = data["std_dev"].apply(lambda row: row ** 2)
    variance_stats = _mean_sem_of_2_grouped_by_1(data, modes)
    return pl.bar(
        x=modes,
        y=variance_stats["mean"],
        error=variance_stats["sem"],
        title=f"{dataset}: ChatGPT responses, variance by mode",
        axis_labels=("ChatGPT mode", "mean of variances (SEM)"),
    )


def training_sq_mode_dists_mode_fig(
    data: pd.DataFrame,
    dataset: str,
) -> Figure:
    modes = range(1,11)
    variance_stats = _mean_sem_of_2_grouped_by_1(pd.DataFrame(pd.concat((
        data["mode"],
        data.filter(like="o").apply(_sq_mode_dist, axis=1),
    ), axis=1)), modes)
    return pl.bar(
        x=modes,
        y=variance_stats["mean"],
        error=variance_stats["sem"],
        title=f"{dataset}: ChatGPT responses, expected squared distance to mode by mode",
        axis_labels=("ChatGPT mode", "mean of expected squared distance to mode (SEM)"),
    )


def training_means_mode_fig(
    data: pd.DataFrame,
    dataset: str,
) -> Figure:
    modes = range(1,11)
    mean_stats = _mean_sem_of_2_grouped_by_1(data, modes)
    return pl.bar(
        x=modes,
        y=mean_stats["mean"],
        error=mean_stats["sem"],
        title=f"{dataset}: ChatGPT responses, mean by mode",
        axis_labels=("ChatGPT mode", "mean of means (SEM)"),
    )


def training_mode_agreement_mode_fig(
    data: pd.DataFrame,
    dataset: str,
) -> Figure:
    modes = range(1,11)
    agreement = data.apply(lambda row: _agreement(row, "mode"), axis=1)
    agreement_stats = _mean_sem_of_2_grouped_by_1(pd.concat((data["mode"], agreement), axis=1), modes)
    return pl.bar(
        x=modes,
        y=agreement_stats["mean"],
        error=agreement_stats["sem"],
        title=f"{dataset}: ChatGPT responses, agreement with mode by mode",
        axis_labels=("ChatGPT mode", "mean of means (SEM)"),
    )


def training_n_mode_fig(
    data: pd.DataFrame,
    dataset: str,
) -> Figure:
    modes = range(1,11)
    ns = data.groupby("mode").agg(n=("mode", "count")).reindex(modes).fillna(0.0)
    return pl.bar(
        x=modes,
        y=ns["n"],
        title=f"{dataset}: ChatGPT responses, number of responses by mode",
        axis_labels=("ChatGPT mode", "n"),
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
    dataset = ds.Dataset.from_csv(
        cfg.dataset_file,
        "no_match!@#",
        ["mean", "mode", "std_dev", "prob"],
    )
    responses = pd.read_csv(cfg.responses_file)
    means = pd.concat([pd.Series(dataset.y["mean"])] * responses.shape[1], axis=1)
    std_devs = pd.concat([pd.Series(dataset.y["std_dev"])] * responses.shape[1], axis=1)
    means.columns = responses.columns
    std_devs.columns = responses.columns
    training_variances_fig(
        dataset.y["mean"],
        dataset.y["std_dev"],
        cfg.dataset,
    ).savefig(cfg.results_dir / "00_training_variances")
    training_sq_mode_dists_fig(
        dataset.y["mean"],
        pd.Series(dataset.y["mode"]),
        dataset.y.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.results_dir / "01_training_sq_mode_dists")
    training_mode_fig(
        dataset.y["mean"],
        dataset.y["mode"],
        cfg.dataset,
    ).savefig(cfg.results_dir / "02_training_modes")
    # training_confidence_fig(
    #     dataset.y["mean"],
    #     dataset.y["std_dev"],
    #     cfg.dataset,
    # ).savefig(cfg.results_dir / "02_training_confidences")
    training_mean_agreement_fig(
        pd.Series(dataset.y["mean"]),
        dataset.y.filter(like="prob"),
        cfg.dataset,
    ).savefig(cfg.results_dir / "03_training_mean_agreements")
    training_variances_mode_fig(
        dataset.y.filter(("mode", "std_dev")),
        cfg.dataset,
    ).savefig(cfg.results_dir / "10_training_variances_mode")
    training_sq_mode_dists_mode_fig(
        dataset.y.filter(like="o"),
        cfg.dataset,
    ).savefig(cfg.results_dir / "11_training_sq_mode_dists_mode")
    training_means_mode_fig(
        dataset.y.filter(("mode", "mean")),
        cfg.dataset,
    ).savefig(cfg.results_dir / "12_training_means_mode")
    training_mode_agreement_mode_fig(
        dataset.y.filter(like="o"),
        cfg.dataset,
    ).savefig(cfg.results_dir / "13_training_mode_agreements")
    training_n_mode_fig(
        dataset.y.filter(like="mode"),
        cfg.dataset,
    ).savefig(cfg.results_dir / "14_training_mode_ns")
    responses_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "20_responses")
    abs_errors_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "30_abs_errors")
    squared_errors_fig(responses, means, cfg.dataset).savefig(cfg.results_dir / "31_squared_errors")
    zscores_fig(responses, means, std_devs, cfg.dataset).savefig(cfg.results_dir / "32_zscores")


def main():
    cfg = read_config("config.json")
    if cfg.retrieve_runs:
        retrieve_runs(cfg)
    if cfg.make_plots:
        make_plots(cfg)


if __name__ == "__main__":
    main()

