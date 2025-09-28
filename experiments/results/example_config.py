from pathlib import Path


DATASET = "smallville_846"
DIRECTORY = Path(DATASET)
DATASET_PATH = (Path("../tpot/data") / DATASET).with_suffix(".csv")
OUTPUTS_PATHS = [
    Path("../tpot/outputs") / DATASET,
]
RUN_ID_PATTERN = "*07-16*/pipeline_data.json"
SUMMARY = DIRECTORY / "summary.csv"
RESPONSES = DIRECTORY / "responses.csv"

