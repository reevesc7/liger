# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Literal, Self, SupportsIndex
from dataclasses import dataclass, field, fields
import warnings
import logging
from pathlib import Path
import shutil
from functools import cached_property
import json
from random import randint
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import liger.training.core as training
from liger.dataset import Dataset
from liger.serde import core as serde, tpot as tpserde
from tpot import TPOTEstimator, Population
from tpot.search_spaces.base import SearchSpace
from tpot.selectors.nsgaii import survival_select_NSGA2
from tpot.selectors.tournament_selection_dominated import tournament_selection_dominated
from joblib import Memory
from distributed import Client
import dill


warnings.filterwarnings(
    "ignore",
    message="The hashes produced for directed graphs changed in version v3.5",
)
warnings.filterwarnings(
    "ignore",
    message=(
        "The behavior of DataFrame concatenation with "
        "empty or all-NA entries is deprecated."
    ),
)


class _TruthyInt(int):
    """Part of a fix for an off-by-one error in TPOT's early stop logic.
    Always returning true allows TPOT's early stop check to occur when
    `TPOTEstimator.early_stop = 0`, allowing for early stopping in the
    case of 2 generations in a row with the same historical top scores.
    """
    def __bool__(self) -> bool:
        return True


class TPOTFitException(Exception):
    """Raised when `TPOTEstimator.fit()` raises an exception
    within `TPOTManager.run_segment()`
    """
    pass


@dataclass
class TPOTManager:
    output_dir: Path
    search_space: dict[str, Any]
    scorers: list[str | Callable]
    scorers_weights: list[float]
    classification: bool
    data_file: Path
    feature_keys: list[str]
    target_keys: list[str]
    feature_transformers: list[str] | None
    target_transformers: list[str] | None
    feature_transformers_kwargs: list[dict[str, Any] | None] | None
    target_transformers_kwargs: list[dict[str, Any] | None] | None
    cv: int
    other_objective_functions: list[Callable]
    other_objective_functions_weights: list[float]
    objective_function_names: list | None
    bigger_is_better: bool
    export_graphpipeline: bool
    memory: Literal["auto"] | Path | Memory | None
    categorical_features: list[str] | list[SupportsIndex] | None
    preprocessing: bool
    population_size: int
    initial_population_size: int | None
    population_scaling: float
    generations_until_end_population: int
    segment_generations: int | None
    total_generations: int | None
    segment_time: float | None
    total_time: float | None
    eval_time: float | None
    validation_strategy: Literal["auto", "reshuffled", "split", "none"]
    validation_fraction: float
    disable_label_encoder: bool
    early_stop: int | None
    scorers_early_stop_tol: list[float]
    other_objectives_early_stop_tol: list[float]
    threshold_evaluation_pruning: tuple[float, float] | None
    threshold_evaluation_scaling: float
    selection_evaluation_pruning: tuple[float, float] | None
    selection_evaluation_scaling: float
    min_history_threshold: int
    survival_percentage: float
    crossover_probability: float
    mutate_probability: float
    mutate_then_crossover_probability: float
    crossover_then_mutate_probability: float
    survival_selector: Callable
    parent_selector: Callable
    budget_range: tuple[float, float] | None
    budget_scaling: float
    generations_until_end_budget: int
    stepwise_steps: int
    n_jobs: int
    memory_limit: str | float | None
    client: Client | None
    processes: bool
    verbose: int
    scatter: bool
    random_state: int | None
    export_fitted_pipeline: bool
    generation_scores_: pd.DataFrame = field(
        default_factory=pd.DataFrame,
        metadata={"state": True},
    )
    segment_durations_: list[float] = field(
        default_factory=list,
        metadata={"state": True},
    )
    _objectives_signs: np.ndarray = field(default_factory=lambda: np.ndarray([]))
    _objectives_tols: np.ndarray = field(default_factory=lambda: np.ndarray([]))

    DATETIME_FMT = "%Y-%m-%d_%H-%M-%S.%f"
    CONFIG_NAME = Path("config.json")
    PARAMS_NAME = Path("params.json")
    STATE_NAME = Path("state.json")
    POPULATION_NAME = Path("population.pkl")
    TMP_POPULATION_NAME = Path("population.pkl.tmp")
    FITTED_PIPELINE_NAME = Path("fitted_pipeline.pkl")

    def __init__(
        self,
        output_dir: str | Path,
        search_space: dict[str, Any],
        scorers: list[str | Callable],
        scorers_weights: list[float],
        classification: bool,
        data_file: str | Path,
        feature_keys: list[str],
        target_keys: list[str],
        feature_transformers: list[str] | None = None,
        target_transformers: list[str] | None = None,
        feature_transformers_kwargs: list[dict[str, Any] | None] | None = None,
        target_transformers_kwargs: list[dict[str, Any] | None] | None = None,
        cv: int = 10,
        other_objective_functions: list[Callable] = [],
        other_objective_functions_weights: list[float] = [],
        objective_function_names: list | None = None,
        bigger_is_better: bool = True,
        export_graphpipeline: bool = False,
        memory: Literal["auto"] | str | Path | Memory | None = None,
        categorical_features: list[str] | list[SupportsIndex] | None = None,
        preprocessing: bool = False,
        population_size: int = 50,
        initial_population_size: int | None = None,
        population_scaling: float = 0.5,
        generations_until_end_population: int = 1,
        segment_generations: int | None = None,
        total_generations: int | None = None,
        segment_time: float | None = 60.0,
        total_time: float | None = 600.0,
        eval_time: float | None = 10.0,
        validation_strategy: Literal["auto", "reshuffled", "split", "none"] = "none",
        validation_fraction: float = 0.2,
        disable_label_encoder: bool = False,
        early_stop: int | None = None,
        scorers_early_stop_tol: float | list[float | None] | None = 0.001,
        other_objectives_early_stop_tol: float | list[float | None] | None = None,
        threshold_evaluation_pruning: tuple[float, float] | None = None,
        threshold_evaluation_scaling: float = 0.5,
        selection_evaluation_pruning: tuple[float, float] | None = None,
        selection_evaluation_scaling: float = 0.5,
        min_history_threshold: int = 0,
        survival_percentage: float = 1.0,
        crossover_probability: float = 0.2,
        mutate_probability: float = 0.7,
        mutate_then_crossover_probability: float = 0.05,
        crossover_then_mutate_probability: float = 0.05,
        survival_selector: Callable = survival_select_NSGA2,
        parent_selector: Callable = tournament_selection_dominated,
        budget_range: tuple[float, float] | None = None,
        budget_scaling: float = 0.5,
        generations_until_end_budget: int = 1,
        stepwise_steps: int = 1,
        n_jobs: int = 1,
        memory_limit: str | float | None = "auto",
        client: Client | None = None,
        processes: bool = True,
        verbose: int = 1,
        scatter: bool = True,
        random_state: Literal["auto"] | int | None = None,
        export_fitted_pipeline: bool = False,
    ) -> None:
        if segment_time == float("inf"):
            segment_time = None
        if total_time == float("inf"):
            total_time = None
        if eval_time == float("inf"):
            eval_time = None
        if segment_generations is None:
            segment_generations = total_generations
        if segment_time is None:
            segment_time = total_time
        if isinstance(memory, Path) or (isinstance(memory, str) and memory != "auto"):
            memory = Path(memory).resolve()
        if random_state == "auto":
            random_state = randint(0, 2 ** 32 - 1)
        self.output_dir = Path(output_dir).resolve()
        self.search_space = search_space
        self.scorers = scorers
        self.scorers_weights = scorers_weights
        self.classification = classification
        self.data_file = Path(data_file).resolve()
        self.feature_keys = feature_keys
        self.target_keys = target_keys
        self.feature_transformers = feature_transformers
        self.target_transformers = target_transformers
        self.feature_transformers_kwargs = feature_transformers_kwargs
        self.target_transformers_kwargs = target_transformers_kwargs
        self.cv = cv
        self.other_objective_functions = other_objective_functions
        self.other_objective_functions_weights = other_objective_functions_weights
        self.objective_function_names = objective_function_names
        self.bigger_is_better = bigger_is_better
        self.export_graphpipeline = export_graphpipeline
        self.memory = memory
        self.categorical_features = categorical_features
        self.preprocessing = preprocessing
        self.population_size = population_size
        self.initial_population_size = initial_population_size
        self.population_scaling = population_scaling
        self.generations_until_end_population = generations_until_end_population
        self.segment_generations = segment_generations
        self.total_generations = total_generations
        self.segment_time = segment_time
        self.total_time = total_time
        self.eval_time = eval_time
        self.validation_strategy = validation_strategy
        self.validation_fraction = validation_fraction
        self.disable_label_encoder = disable_label_encoder
        self.early_stop = early_stop
        self.scorers_early_stop_tol = self._init_tol(
            len(scorers),
            scorers_early_stop_tol,
        )
        self.other_objectives_early_stop_tol = self._init_tol(
            len(other_objective_functions),
            other_objectives_early_stop_tol,
        )
        self.threshold_evaluation_pruning = threshold_evaluation_pruning
        self.threshold_evaluation_scaling = threshold_evaluation_scaling
        self.selection_evaluation_pruning = selection_evaluation_pruning
        self.selection_evaluation_scaling = selection_evaluation_scaling
        self.min_history_threshold = min_history_threshold
        self.survival_percentage = survival_percentage
        self.crossover_probability = crossover_probability
        self.mutate_probability = mutate_probability
        self.mutate_then_crossover_probability = mutate_then_crossover_probability
        self.crossover_then_mutate_probability = crossover_then_mutate_probability
        self.survival_selector = survival_selector
        self.parent_selector = parent_selector
        self.budget_range = budget_range
        self.budget_scaling = budget_scaling
        self.generations_until_end_budget = generations_until_end_budget
        self.stepwise_steps = stepwise_steps
        self.n_jobs = n_jobs
        self.memory_limit = memory_limit
        self.client = client
        self.processes = processes
        self.verbose = verbose
        self.scatter = scatter
        self.random_state = random_state
        self.export_fitted_pipeline = export_fitted_pipeline
        self.generation_scores_ = pd.DataFrame()
        self.segment_durations_ = []
        self._objectives_signs = np.sign(
            self.scorers_weights + self.other_objective_functions_weights
        ) * (1 if self.bigger_is_better else -1)
        self._objectives_tols = np.array(
            self.scorers_early_stop_tol + self.other_objectives_early_stop_tol
        )
        self._check_end_conditions()

    @staticmethod
    def _init_tol(
        n_tols: int,
        tol_val: float | list[float | None] | None,
    ) -> list[float]:
        if not isinstance(tol_val, list):
            tol_vals = [tol_val for _ in range(n_tols)]
        else:
            tol_vals = tol_val
        return [float("inf") if tol is None else tol for tol in tol_vals]

    def _is_run_endless(self) -> bool:
        return (
            self.total_generations is None
            and
            self.total_time is None
            and
            self.early_stop is None
            and
            all(tol is None for tol in self._objectives_tols)
        )

    def _log(
        self,
        message: str,
        category: int | type[Warning] = logging.INFO,
    ) -> None:
        if isinstance(category, type):
            log_level = logging.WARN
        else:
            log_level = category
            category = UserWarning
        if log_level < logging.ERROR - 10 * self.verbose:
            return
        if log_level >= logging.WARN:
            warnings.warn(message, category, stacklevel=2)
            return
        print(message)

    def _check_end_conditions(self) -> None:
        endless_segment = self.segment_generations is None and self.segment_time is None
        endless_run = self._is_run_endless()
        if endless_segment and not endless_run:
            self._log((
                "No segment end condition is set; "
                    "any segment will continue until run end condition is reached"
            ), logging.WARN)
        if not endless_segment and endless_run:
            self._log((
                "No run end condition has been set; each segment will continue until "
                    "segment end condition is reached; ! - run will never register as "
                    "complete for the purposes of automated segment running - !"
            ), logging.WARN)
        if endless_segment and endless_run:
            self._log((
                "!!! - No segment or run end conditions set; "
                    "any segment will continue indefinitely - !!!"
            ), logging.WARN)

    @staticmethod
    def _parse_config(config_path: str | Path) -> dict[str, Any]:
        config_path = Path(config_path)
        params = json.load(config_path.open())
        if not isinstance(params, dict):
            raise ValueError(f"{config_path} does not contain a JSON object")
        if "scorers" in params:
            parsed_scorers = []
            for scorer in params["scorers"]:
                if isinstance(scorer, str):
                    if "." not in scorer:
                        parsed_scorers.append(scorer)
                        continue
                    parsed_scorers.append(serde.import_from_str(scorer))
                    continue
                if isinstance(scorer, dict):
                    factory: Callable = serde.import_from_str(scorer["factory"])
                    parsed_scorers.append(factory(
                        serde.import_from_str(scorer["kwargs"]["score_func"]),
                        scorer["kwargs"]["sign"],
                        scorer["kwargs"]["kwargs"],
                        scorer["kwargs"]["response_method"],
                    ))
            params["scorers"] = parsed_scorers
            # params["scorers"] = [
            #     serde.import_from_str(scorer) if "." in scorer else scorer
            #     for scorer in params["scorers"]
            # ]
        if "other_objective_functions" in params:
            params["other_objective_functions"] = [
                serde.import_from_str(func)
                for func in params["other_objective_functions"]
            ]
        if "survival_selector" in params:
            params["survival_selector"] = serde.import_from_str(
                params["survival_selector"]
            )
        if "parent_selector" in params:
            params["parent_selector"] = serde.import_from_str(
                params["parent_selector"]
            )
        return params

    @classmethod
    def from_config(cls, config_path: str | Path, **kwargs: Any) -> Self:
        config_path = Path(config_path)
        params = cls._parse_config(config_path)
        params.update(kwargs)
        if "output_dir" not in kwargs:
            params["output_dir"] = Path(
                params.get("output_dir", "")
            ) / datetime.now(timezone.utc).strftime(cls.DATETIME_FMT)
        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_path, output_dir / cls.CONFIG_NAME)
        inst = cls(**params)
        inst.export_params()
        return inst

    @staticmethod
    def _path_default(path: str | Path | None, default: Path) -> Path:
        if path is None:
            return default
        return Path(path)

    def import_state(self, filepath: str | Path | None = None) -> None:
        filepath = self._path_default(filepath, self.output_dir / self.STATE_NAME)
        self._log(f"Importing state from {filepath}", logging.DEBUG)
        state = json.load(filepath.open())
        if not isinstance(state, dict):
            raise ValueError(f"{filepath} does not contain a JSON object")
        for fld in fields(self):
            if not fld.metadata.get("state", False):
                continue
            value = state.get(fld.name)
            if value is not None:
                setattr(self, fld.name, value)
        self.generation_scores_ = pd.DataFrame(self.generation_scores_)
        self.generation_scores_.index = self.generation_scores_.index.astype(np.int64)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> Self:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        params = cls._parse_config(checkpoint_dir / cls.PARAMS_NAME)
        params["output_dir"] = checkpoint_dir
        inst = cls(**params)
        if (checkpoint_dir / cls.STATE_NAME).is_file():
            inst.import_state(checkpoint_dir / cls.STATE_NAME)
        return inst

    @cached_property
    def _dataset(self) -> Dataset:
        return Dataset.from_csv(
            self.data_file,
            self.feature_keys,
            self.target_keys,
            self.feature_transformers,
            self.target_transformers,
            self.feature_transformers_kwargs,
            self.target_transformers_kwargs,
        )

    @cached_property
    def _search_space(self) -> SearchSpace:
        parser = tpserde.SearchSpaceParser(
            self._dataset.x.shape[0],
            self._dataset.y.shape[0],
            self.random_state,
        )
        return parser.parse(self.search_space)

    def safe_cv(self) -> int:
        if self.classification:
            _, counts = np.unique(self._dataset.y, return_counts=True)
            if counts.size == 1:
                max_cv = int(counts[0])
            else:
                max_cv = int(np.sort(counts)[-2])
        else:
            max_cv = self._dataset.y.shape[0]
        if self.cv > max_cv:
            self._log((
                f"Provided cv value {self.cv!r} is greater than the dataset allows; "
                    f"using max allowed by dataset: {max_cv!r}"
            ), logging.WARN)
            return max_cv
        return self.cv

    @cached_property
    def _tpot(self) -> TPOTEstimator:
        # Corrects for off-by-one error in TPOT's early_stop logic
        if isinstance(self.early_stop, int):
            early_stop = _TruthyInt(self.early_stop - 1)
        else:
            early_stop = self.early_stop
        return TPOTEstimator(
            search_space=self._search_space,
            scorers=self.scorers,
            scorers_weights=self.scorers_weights,
            classification=self.classification,
            cv=self.safe_cv(),
            other_objective_functions=self.other_objective_functions,
            other_objective_functions_weights=self.other_objective_functions_weights,
            objective_function_names=self.objective_function_names,
            bigger_is_better=self.bigger_is_better,
            export_graphpipeline=self.export_graphpipeline,
            memory=self.memory,
            categorical_features=self.categorical_features,
            preprocessing=self.preprocessing,
            population_size=self.population_size,
            initial_population_size=self.initial_population_size,
            population_scaling=self.population_scaling,
            generations_until_end_population=self.generations_until_end_population,
            generations=self.segment_generations,
            max_time_mins=self.segment_time, # pyright: ignore [reportArgumentType]
            max_eval_time_mins=self.eval_time, # pyright: ignore [reportArgumentType]
            validation_strategy=self.validation_strategy,
            validation_fraction=self.validation_fraction,
            disable_label_encoder=self.disable_label_encoder,
            early_stop=early_stop,
            scorers_early_stop_tol=self.scorers_early_stop_tol, # pyright: ignore [reportArgumentType]
            other_objectives_early_stop_tol=self.other_objectives_early_stop_tol,
            threshold_evaluation_pruning=self.threshold_evaluation_pruning,
            threshold_evaluation_scaling=self.threshold_evaluation_scaling,
            selection_evaluation_pruning=self.selection_evaluation_pruning,
            selection_evaluation_scaling=self.selection_evaluation_scaling,
            min_history_threshold=self.min_history_threshold,
            survival_percentage=self.survival_percentage, # pyright: ignore [reportArgumentType]
            crossover_probability=self.crossover_probability,
            mutate_probability=self.mutate_probability,
            mutate_then_crossover_probability=self.mutate_then_crossover_probability,
            crossover_then_mutate_probability=self.crossover_then_mutate_probability,
            survival_selector=self.survival_selector,
            parent_selector=self.parent_selector,
            budget_range=self.budget_range,
            budget_scaling=self.budget_scaling,
            generations_until_end_budget=self.generations_until_end_budget,
            stepwise_steps=self.stepwise_steps,
            n_jobs=self.n_jobs,
            memory_limit=self.memory_limit,
            client=self.client,
            processes=self.processes,
            warm_start=True,
            periodic_checkpoint_folder=self.output_dir,
            verbose=self.verbose,
            scatter=self.scatter,
            random_state=self.random_state,
        )

    def export_params(self, filepath: str | Path | None = None) -> None:
        filepath = self._path_default(filepath, self.output_dir / self.PARAMS_NAME)
        self._log(f"Exporting parameters to {filepath}", logging.DEBUG)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        params = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if not (fld.name.startswith("_") or fld.name.endswith("_"))
        }
        json.dump(
            params,
            filepath.open("w"),
            indent=4,
            default=serde.to_json_compatible,
        )

    def _restore_population_from_backup(self) -> None:
        """Restore population file from the temporary backup if it exists.
        #
        Called in case an improperly terminated segment, leaving
        a population file with a partially complete last generation.
        #
        Population file is deleted.
        If a backup exists it is moved to the path population file path.
        """
        self._log("\n-- Restoring population file from backup --\n")
        (self.output_dir / self.POPULATION_NAME).unlink(missing_ok=True)
        if (self.output_dir / self.TMP_POPULATION_NAME).is_file():
            (self.output_dir / self.TMP_POPULATION_NAME).replace(
                self.output_dir / self.POPULATION_NAME
            )

    def _open_segment(self) -> None:
        if (self.output_dir / self.TMP_POPULATION_NAME).is_file():
            self._log(
                "Found temporary population file; "
                    "assuming that previous segment finished improperly; "
                    "restoring state of previous segment and re-running"
            )
            self._restore_population_from_backup()
        if (self.output_dir / self.POPULATION_NAME).is_file():
            shutil.copy(
                self.output_dir / self.POPULATION_NAME,
                self.output_dir / self.TMP_POPULATION_NAME,
            )

    @property
    def complete_generations(self) -> int:
        max_generation = self.generation_scores_.index.max()
        if isinstance(max_generation, np.integer):
            return max_generation + 1
        return 0

    def _is_end_generation(self, log: bool = False) -> bool:
        if self.total_generations is None:
            return False
        if self.complete_generations < self.total_generations:
            return False
        if log:
            self._log((
                f"Complete generations, {self.complete_generations}, "
                    f"exceeds total generations, {self.total_generations}"
            ), logging.DEBUG)
        return True

    def _is_end_time(self, log: bool = False) -> bool:
        if self.total_time is None:
            return False
        if sum(self.segment_durations_) < 60.0 * self.total_time:
            return False
        if log:
            self._log((
                f"Sum of segment durations, {sum(self.segment_durations_) / 60.0}, "
                    f"exceeds total time, {self.total_time}"
            ), logging.DEBUG)
        return True

    def _is_end_early_stop(self, log: bool = False) -> bool:
        """Detect whether the early stop condition has been met.
        #
        Triggers if there has not been improvement between
        any of the last `early_stop` pairs of generations.
        I.e., scores must be the same for `early_stop + 1` generations.
        """
        # TPOT checks that the budget is at max before evaluating early stop.
        # Max budget is generally reached on the generation specified in end_budget,
        # TPOT checks whether the maximum scores across all previously
        # evaluated individuals increased from those of the last generation.
        # TPOT counts how many generations have not improved over each previous.
        if self.early_stop is None:
            return False
        if self.complete_generations <= self.early_stop:
            return False
        if self.budget_range is not None and self.complete_generations <= (
            self.generations_until_end_budget + self.early_stop - 1
        ):
            return False
        big_scores = (
            self.generation_scores_.to_numpy() * self._objectives_signs
        )
        gens_top_scores = np.maximum.accumulate(big_scores)[-self.early_stop - 1:]
        if (np.diff(gens_top_scores, axis=0) > self._objectives_tols).any():
            return False
        if log:
            self._log((
                "Scores within early stop tolerances across "
                    f"{self.early_stop + 1} generations"
            ), logging.DEBUG)
        return True

    def is_complete(self, log: bool = False) -> bool:
        if self._is_run_endless():
            if log:
                self._log("No end condition set, run cannot be complete", logging.DEBUG)
            return False
        return any((
            self._is_end_generation(log),
            self._is_end_time(log),
            self._is_end_early_stop(log),
        ))

    @staticmethod
    def _fix_individuals(evaluated_individuals: pd.DataFrame) -> pd.DataFrame:
        evaluated_individuals["Generation"] = (
            evaluated_individuals["Generation"].astype("Int64")
        )
        return evaluated_individuals

    @property
    def _fixed_individuals(self) -> pd.DataFrame | None:
        if self._tpot.evaluated_individuals is None:
            return None
        return self._fix_individuals(self._tpot.evaluated_individuals)

    def _record_generation_scores(self) -> None:
        if self._fixed_individuals is None:
            return
        obj_names = self._tpot.objective_names
        scores = pd.DataFrame(
            self._fixed_individuals[obj_names + ["Generation"]]
        )
        scores["Generation"] = scores["Generation"].astype(int)
        scores.set_index("Generation")
        scores[obj_names] = scores[obj_names] * self._objectives_signs
        top_scores = pd.DataFrame(
            scores.groupby("Generation").max() * self._objectives_signs
        )
        self.generation_scores_ = pd.DataFrame(top_scores[obj_names])

    def export_state(
        self,
        evaluated_individuals: pd.DataFrame | None = None,
        filepath: str | Path | None = None,
    ) -> None:
        filepath = self._path_default(filepath, self.output_dir / self.STATE_NAME)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.metadata.get("state", False)
        } | {"complete_": self.is_complete()}
        if evaluated_individuals is not None:
            state.update({"evaluated_individuals": evaluated_individuals.drop(
                "Individual",
                axis=1,
            )})
        json.dump(
            state,
            filepath.open("w"),
            indent=4,
            default=serde.to_json_compatible,
        )

    def _load_population(self, filepath: str | Path | None = None) -> Population:
        filepath = self._path_default(filepath, self.output_dir / self.POPULATION_NAME)
        population = dill.load(filepath.open("rb"))
        if not isinstance(population, Population):
            raise TypeError(f"{filepath} does not contain a Population object")
        return population

    def _load_fix_population(self, filepath: str | Path | None = None) -> Population:
        population = self._load_population(filepath)
        population.evaluated_individuals = self._fix_individuals(
            population.evaluated_individuals
        )
        return population

    def _dump_population(
        self,
        population: Population,
        filepath: str | Path | None = None,
    ) -> None:
        filepath = self._path_default(filepath, self.output_dir / self.POPULATION_NAME)
        dill.dump(population, filepath.open("wb"))

    def export_pipeline(self, filepath: str | Path | None = None) -> None:
        """Export a pickle file of the best performing pipeline.
        #
        Pickling and unplicking pipelines relies on the `dill` module.
        #
        Parameters
        ----------
        `filepath` : `str | Path`, optional
            Path to write to. Writes to the output directory by default.
        """
        filepath = self._path_default(
            filepath,
            self.output_dir / self.FITTED_PIPELINE_NAME
        )
        dill.dump(self._tpot.fitted_pipeline_, filepath.open("wb"))

    def _run_segment(self) -> None:
        start_time = time.time()
        if self.is_complete():
            self._log("\nRun end conditions already met; aborting")
            return
        self._log(f"\nRun output directory: {self.output_dir}\n")
        try:
            self._tpot.fit(self._dataset.x, self._dataset.y)
        except (KeyboardInterrupt, SystemExit):
            self._restore_population_from_backup()
            raise
        except Exception as e:
            self._restore_population_from_backup()
            raise TPOTFitException(
                "Encountered an exception during TPOTEstimator.fit()"
            ) from e
        self._record_generation_scores()
        self.segment_durations_.append(time.time() - start_time)
        self.export_state(self._fixed_individuals)
        self._dump_population(self._load_fix_population())
        if not self.is_complete(log=True):
            self._log("\nEnd of segment\nRun incomplete\n")
            return
        if self.export_fitted_pipeline:
            self.export_pipeline()
        self._log(f"\n{self._tpot.fitted_pipeline_}")
        self._log("\nEnd of segment\nRun complete\n")

    def _close_segment(self) -> None:
        (self.output_dir / self.TMP_POPULATION_NAME).unlink(missing_ok=True)

    def run_segment(self) -> None:
        self._open_segment()
        with training.sigterm_handler():
            try:
                self._run_segment()
            except training.SIGTERMReceived as e:
                self._restore_population_from_backup()
                e.args = ("SIGTERM received",)
                raise
            self._close_segment()
