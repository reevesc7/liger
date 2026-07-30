from pathlib import Path
import json
from liger.results_processing.json_cache import JSONCache


INPUT_DIR = Path("../../../runs_data/e03/outputs")


def key_name(key: str) -> str:
    if key == "id":
        return "run_id"
    if key == "target_gens":
        return "total_generations"
    return key


def main():
    cache = JSONCache(
        INPUT_DIR,
        "manager_data.json",
    )
    for run_path, run_data in cache:
        new_path = run_path.parent / "run_data.json"
        new_data = {
            key_name(key): value
            for key, value in run_data["manager_parameters"].items()
            if key != "eval_random_states"
        }
        new_data.update(run_data["tpot_parameters"])
        new_data.update(run_data["manager_attributes"])
        new_data["evaluated_individuals"] = run_data["tpot_attributes"][
            "evaluated_individuals"
        ]
        with open(new_path, "w") as file:
            json.dump(new_data, file, indent=4)


if __name__ == "__main__":
    main()
