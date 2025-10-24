import pandas as pd
import liger.output_processing as op
#from experiments.results.config import DIRECTORY, OUTPUTS_PATHS, RUN_ID_PATTERN
import experiments.results.config as cfg


def find_root(fitted_pipeline: list[str]) -> str:
    """Only works with GraphPipeline runs
    """
    string = fitted_pipeline[0]
    start = string.find("'") + 1
    end = string.find("_", start)
    return string[start:end]


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


def main():
    data = op.mass_json_load(paths=cfg.OUTPUTS_PATHS, pattern=cfg.RUN_ID_PATTERN)
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
            #responses[summary["id"][-1]] = order_responses(kfold_predictions)
            responses.update(run_responses(summary["id"][-1], kfold_predictions))
    cfg.DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary)
    print(summary)
    summary.to_csv(cfg.DIRECTORY / "summary.csv", index=False)
    responses = pd.DataFrame(responses)
    print(responses)
    responses.to_csv(cfg.DIRECTORY / "responses.csv", index=False)


if __name__ == "__main__":
    main()

