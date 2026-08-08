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


from typing import Callable, ClassVar, Iterable, Protocol
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path
import json
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import dill
from scipy.sparse import spmatrix
from sklearn.utils.multiclass import type_of_target
from tpot import TPOTEstimator, Population
from tpot.search_spaces.base import SearchSpace
from tpot.utils import beta_interpolation
import liger.config as cfg
from ._params import (
    Objective,
    InverseObjectives,
    DatasetParams,
    EvolutionParams,
    EvalParams,
    EndParams,
    RuntimeParams,
)


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


class SearchSpaceInitializer(Protocol):
    def __call__(
        self,
        *,
        n_classes: int,
        n_samples: int,
        n_features: int,
        random_state: int | None,
    ) -> SearchSpace: ...


class _TruthyInt(int):
    """Part of a fix for an off-by-one error in TPOT's early stop logic.
    Always returning true allows TPOT's early stop check to occur when
    `TPOTEstimator.early_stop = 0`, allowing for early stopping in the
    case of 2 generations in a row with the same historical top scores.
    """
    def __bool__(self) -> bool:
        return True


@dataclass(slots=True)
class TPOTManager:
    POP_NAME: ClassVar[str] = "population.pkl"
    INDIVS_NAME: ClassVar[str] = "individuals.csv"
    INSTANCES_NAME: ClassVar[str] = "instances.json"
    ID_INDEX_NAME: ClassVar[str] = "ID"

    output_dir: str | Path
    output_dir_: Path = field(init=False)
    x: ArrayLike | spmatrix | Callable[[], ArrayLike | spmatrix]
    _x: np.ndarray | pd.DataFrame | spmatrix | None = field(default=None, init=False)
    y: ArrayLike | spmatrix | Callable[[], ArrayLike | spmatrix]
    _y: np.ndarray | pd.DataFrame | spmatrix | None = field(default=None, init=False)
    search_space: str | SearchSpace | SearchSpaceInitializer
    _search_space: str | SearchSpace | None = field(default=None, init=False)
    classification: bool
    objectives: Iterable[Objective]
    _objectives: InverseObjectives = field(init=False)
    dataset_params: DatasetParams = field(default_factory=DatasetParams)
    evolution_params: EvolutionParams = field(default_factory=EvolutionParams)
    eval_params: EvalParams = field(default_factory=EvalParams)
    end_params: EndParams = field(default_factory=EndParams)
    runtime_params: RuntimeParams = field(default_factory=RuntimeParams)
    verbose: int = 0
    is_complete_: bool | None = field(default=None, init=False)
    _tpot: TPOTEstimator | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output_dir = self.output_dir_ = Path(self.output_dir).resolve()
        self._objectives = InverseObjectives(self.objectives)
        self._check_end_conditions()

    @property
    def pop_path_(self) -> Path:
        return self.output_dir_ / self.POP_NAME

    @property
    def indivs_path_(self) -> Path:
        return self.output_dir_ / self.INDIVS_NAME

    @property
    def instances_path_(self) -> Path:
        return self.output_dir_ / self.INSTANCES_NAME

    def _log(self, message: str, category: int | type[Warning] = logging.INFO) -> None:
        if isinstance(category, type):
            log_level = logging.WARNING
        else:
            log_level = category
            category = UserWarning
        if log_level < logging.ERROR - 10 * self.verbose:
            return
        if log_level >= logging.WARNING:
            warnings.warn(
                f"{logging.getLevelName(logging.WARNING)}: {message}",
                category,
                stacklevel=2,
            )
            return
        print(f"{logging.getLevelName(log_level)}: {message}")

    def _check_end_conditions(self) -> None:
        endless_segment = self.end_params.segment_is_endless_
        endless_run = self.end_params.run_is_endless_
        if endless_segment and not endless_run:
            self._log((
                "No segment end condition is set; "
                    "any segment will continue until run end condition is reached"
            ), logging.WARNING)
        if not endless_segment and endless_run:
            self._log((
                "No run end condition has been set; each segment will continue until "
                    "segment end condition is reached; ! - run will never register as "
                    "complete for the purposes of automated segment running - !"
            ), logging.WARNING)
        if endless_segment and endless_run:
            self._log((
                "!!! - No segment or run end conditions set; "
                    "any segment will continue indefinitely - !!!"
            ), logging.WARNING)

    def _gen_pop_size(self, gen: int) -> int:
        start_size = self.evolution_params.initial_population_size
        end_size = self.evolution_params.population_size
        if start_size is None or start_size == end_size:
            return end_size
        gen_of_end_size = self.evolution_params.generations_until_end_population
        # /// --- Corrects for 'beta_interpolation()' edge case --- \\\
        if gen_of_end_size <= 1:
            gen_of_end_size = 2
        # \\\ ----------------------------------------------------- ///
        if gen >= gen_of_end_size:
            return end_size
        per_gen_pop_size: list[np.float64] = beta_interpolation(
            start=start_size,
            end=end_size,
            scale=self.evolution_params.population_scaling, # type: ignore
            n=gen_of_end_size,
        )
        return np.round(per_gen_pop_size[gen]).astype(int)

    def _load_fix_pop(self) -> Population | None:
        if not self.pop_path_.is_file():
            self._log(f"No population file at {self.pop_path_!r}", logging.DEBUG)
            return None
        self._log(f"Loading population from {self.pop_path_!r}", logging.DEBUG)
        pop = dill.load(self.pop_path_.open("rb"))
        if not isinstance(pop, Population):
            raise TypeError(f"{self.pop_path_!r} does not contain a "
                f"{Population.__name__!r} object")
        indivs = pop.evaluated_individuals
        if not pd.api.types.is_integer_dtype(indivs["Generation"]):
            indivs["Generation"] = indivs["Generation"].astype("Int64")
        last_gen = self._complete_gens(indivs) - 1
        last_pop_size = self._gen_pop_size(last_gen)
        last_gen_mask = indivs["Generation"] == last_gen
        if last_gen_mask.sum() < last_pop_size:
            self._log("Last generation of population is incomplete; erasing generation")
            if last_gen == 0:
                self.pop_path_.unlink()
                self.indivs_path_.unlink(missing_ok=True)
                return None
            indivs = pd.DataFrame(indivs.loc[~last_gen_mask])
            pop.population = pop.population[:indivs.shape[0]]
        pop.evaluated_individuals = indivs
        return pop

    def _complete_gens(self, indivs: pd.DataFrame) -> int:
        return int(indivs["Generation"].max()) + 1

    def _is_end_gen(self, indivs: pd.DataFrame) -> bool:
        if self.end_params.total_generations is None:
            return False
        complete_gens = self._complete_gens(indivs)
        if complete_gens < self.end_params.total_generations:
            return False
        self._log((
            f"Complete generations, {complete_gens}, meets or exceeds "
                f"total generations, {self.end_params.total_generations}"
        ), logging.DEBUG)
        return True

    def _is_end_time(self, indivs: pd.DataFrame) -> bool:
        if self.end_params.total_time is None:
            return False
        generations = indivs.groupby("Generation")
        gens_elapsed_time = pd.Series(pd.Series(
            generations["Completed Timestamp"].max()
        ) - pd.Series(
            generations["Submitted Timestamp"].min()
        ))
        elapsed_time: np.float64 = gens_elapsed_time.sum()
        if elapsed_time < 60.0 * self.end_params.total_time:
            return False
        self._log((
            f"Sum of segment durations, {elapsed_time / 60.0}, meets or exceeds "
                f"total time, {self.end_params.total_time}"
        ), logging.DEBUG)
        return True

    def _is_end_early_stop(self, indivs: pd.DataFrame) -> bool:
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
        if self.end_params.early_stop is None:
            return False
        complete_gens = self._complete_gens(indivs)
        if complete_gens <= self.end_params.early_stop:
            return False
        if self.eval_params.budget_range is not None and complete_gens <= (
            self.eval_params.generations_until_end_budget
                + self.end_params.early_stop - 1
        ):
            return False
        great_scores = pd.DataFrame(pd.DataFrame(
            indivs.set_index("Generation").iloc[:, :len(self._objectives.signs_)]
        ) * self._objectives.signs_)
        gens_best_scores = great_scores.groupby("Generation").max()
        cum_gens_best_scores = np.maximum.accumulate(
            gens_best_scores
        )[-self.end_params.early_stop - 1:]
        if (
            np.diff(cum_gens_best_scores, axis=0) > self._objectives.early_stop_tols_
        ).any():
            return False
        self._log((
            "Scores within early stop tolerances across "
                f"{self.end_params.early_stop + 1} generations"
        ), logging.DEBUG)
        return True

    def _check_is_complete(self, indivs: pd.DataFrame) -> bool:
        if self.end_params.run_is_endless_:
            self._log("No end condition set, run cannot be complete", logging.DEBUG)
            return False
        if indivs is None or indivs.shape[0] == 0:
            return False
        return any((
            self._is_end_gen(indivs),
            self._is_end_time(indivs),
            self._is_end_early_stop(indivs),
        ))

    def _dump_pop(self, population: Population) -> None:
        self._log(f"Dumping population to {self.pop_path_!r}", logging.DEBUG)
        dill.dump(population, self.pop_path_.open("wb"))

    def _load_indivs(self) -> pd.DataFrame | None:
        if not self.indivs_path_.is_file():
            self._log(f"No individuals file at {self.indivs_path_!r}", logging.DEBUG)
            return None
        self._log(f"Loading individuals from {self.indivs_path_!r}", logging.DEBUG)
        return pd.read_csv(self.indivs_path_, index_col=self.ID_INDEX_NAME)

    def _init_state(self) -> None:
        last_pop = self._load_fix_pop()
        if last_pop is not None:
            self.is_complete_ = self._check_is_complete(last_pop.evaluated_individuals)
            if self.is_complete_:
                return
            self._dump_pop(last_pop)
            return
        last_indivs = self._load_indivs()
        if last_indivs is None:
            self.is_complete_ = False
            return
        self.is_complete_ = self._check_is_complete(last_indivs)
        if not self.is_complete_:
            raise FileNotFoundError(f"Individuals file at {self.indivs_path_!r} "
                "indicates run is incomplete, but no population file exists at "
                f"{self.pop_path_!r}; cannot recover run state")

    def _init_x(self) -> np.ndarray | pd.DataFrame | spmatrix:
        if self._x is not None:
            x = self._x
        elif isinstance(self.x, Callable):
            x = self.x()
        else:
            x = self.x
        if isinstance(x, (np.ndarray, pd.DataFrame, spmatrix)):
            return x
        return np.asarray(x)

    def _init_y(self) -> np.ndarray | pd.DataFrame | spmatrix:
        if self._y is not None:
            y = self._y
        elif isinstance(self.y, Callable):
            y = self.y()
        else:
            y = self.y
        if isinstance(y, (np.ndarray, pd.DataFrame, spmatrix)):
            return y
        return np.asarray(y)

    def _y_class_counts(self) -> tuple[np.ndarray, np.ndarray]:
        if self._y is None:
            self._y = self._init_y()
        y = self._y
        target_type = type_of_target(y)
        if target_type == "multilabel-indicator":
            if not isinstance(y, spmatrix):
                y = np.asarray(y)
            counts = np.asarray(np.ones(y.shape[0], "Int16") @ y).ravel()
            labels = np.arange(y.shape[1])
        else:
            y = np.asarray(self._y).ravel()
            labels, counts = np.unique(y, return_counts=True)
        return labels, counts

    def _init_search_space(self) -> str | SearchSpace:
        if isinstance(self.search_space, (str, SearchSpace)):
            return self.search_space
        if self._x is None:
            self._x = self._init_x()
        return self.search_space(
            n_classes=self._y_class_counts()[0].shape[0],
            n_samples=self._x.shape[0],
            n_features=self._x.shape[1],
            random_state=self.evolution_params.random_state_,
        )

    def _safe_cv(self) -> int:
        if self._y is None:
            self._y = self._init_y()
        if self.classification:
            class_counts = self._y_class_counts()[1]
            if class_counts.size == 1:
                max_cv = int(class_counts[0])
            else:
                max_cv = int(np.sort(class_counts)[-2])
        else:
            max_cv = self._y.shape[0]
        if self.eval_params.cv > max_cv:
            self._log((
                f"Provided cv value {self.eval_params.cv} "
                    "is greater than the dataset allows; "
                    f"using max allowed by dataset: {max_cv}"
            ), logging.WARNING)
            return max_cv
        return self.eval_params.cv

    def _init_tpot(self) -> TPOTEstimator:
        if self._search_space is None:
            self._search_space = self._init_search_space()
        # /// --- Corrects off-by-one error in TPOT's early_stop logic --- \\\
        if isinstance(self.end_params.early_stop, int):
            early_stop = _TruthyInt(self.end_params.early_stop - 1)
        else:
            early_stop = self.end_params.early_stop
        # \\\ ------------------------------------------------------------ ///
        return TPOTEstimator(
            search_space=self._search_space,
            scorers=self._objectives.scorers_,
            scorers_weights=self._objectives.scorers_weights_,
            classification=self.classification,
            cv=self._safe_cv(),
            other_objective_functions=self._objectives.others_,
            other_objective_functions_weights=self._objectives.others_weights_,
            objective_function_names=None,
            bigger_is_better=True,
            export_graphpipeline=False,
            memory=self.runtime_params.make_eval_cache(self.output_dir_),
            categorical_features=self.dataset_params.categorical_features,
            preprocessing=self.dataset_params.preprocessing, # type: ignore
            population_size=self.evolution_params.population_size,
            initial_population_size=self.evolution_params.initial_population_size,
            population_scaling=self.evolution_params.population_scaling,
            generations_until_end_population=(
                self.evolution_params.generations_until_end_population
            ),
            generations=self.end_params.segment_generations,
            max_time_mins=self.end_params.segment_time, # type: ignore
            max_eval_time_mins=self.eval_params.eval_time, # type: ignore
            validation_strategy=self.eval_params.validation_strategy,
            validation_fraction=self.eval_params.validation_fraction,
            disable_label_encoder=self.dataset_params.disable_label_encoder,
            early_stop=early_stop,
            scorers_early_stop_tol=(
                    self._objectives.scorers_early_stop_tols_
            ), # type: ignore
            other_objectives_early_stop_tol=self._objectives.others_early_stop_tols_,
            threshold_evaluation_pruning=self.eval_params.threshold_evaluation_pruning,
            threshold_evaluation_scaling=self.eval_params.threshold_evaluation_scaling,
            selection_evaluation_pruning=self.eval_params.selection_evaluation_pruning,
            selection_evaluation_scaling=self.eval_params.selection_evaluation_scaling,
            min_history_threshold=self.eval_params.min_history_threshold,
            survival_percentage=(
                self.evolution_params.survival_percentage
            ), # type: ignore
            crossover_probability=self.evolution_params.crossover_probability,
            mutate_probability=self.evolution_params.mutate_probability,
            mutate_then_crossover_probability=(
                self.evolution_params.mutate_then_crossover_probability
            ),
            crossover_then_mutate_probability=(
                self.evolution_params.crossover_then_mutate_probability
            ),
            survival_selector=self.evolution_params.survival_selector,
            parent_selector=self.evolution_params.parent_selector,
            budget_range=self.eval_params.budget_range,
            budget_scaling=self.eval_params.budget_scaling,
            generations_until_end_budget=self.eval_params.generations_until_end_budget,
            stepwise_steps=None, # type: ignore
            n_jobs=self.runtime_params.n_jobs,
            memory_limit=self.runtime_params.memory_limit,
            client=self.runtime_params.client,
            processes=self.runtime_params.processes,
            warm_start=True,
            periodic_checkpoint_folder=self.output_dir_,
            verbose=self.verbose,
            scatter=self.runtime_params.scatter,
            random_state=self.evolution_params.random_state_,
        )

    def _dump_indivs(self, indivs: pd.DataFrame) -> None:
        self._log(f"Dumping individuals to {self.indivs_path_!r}", logging.DEBUG)
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        indivs.drop("Individual", axis=1).rename_axis(
            self.ID_INDEX_NAME
        ).to_csv(self.indivs_path_)
        config_indivs = cfg.to_config(indivs["Instance"])
        json.dump(config_indivs, self.instances_path_.open("w"), indent=4)

    def run_segment(self) -> None:
        self._init_state()
        if self.is_complete_:
            self._log("Run end conditions already met; aborting")
            return
        self._log(f"Run output directory: {self.output_dir_!r}")
        if self._tpot is None:
            self._tpot = self._init_tpot()
        if self._x is None:
            self._x = self._init_x()
        if self._y is None:
            self._y = self._init_y()
        self._tpot.fit(self._x, self._y)
        indivs = self._tpot.evaluated_individuals
        if not isinstance(indivs, pd.DataFrame):
            raise TypeError("Evaluated individuals should be type "
                f"{pd.DataFrame.__name__!r} but was type {type(indivs).__name__!r}")
        self._dump_indivs(indivs)
        self.is_complete_ = self._check_is_complete(indivs)
        if not self.is_complete_:
            self._log("End of segment; Run incomplete\n")
            return
        pretty_fitted_pipeline = json.dumps(
            cfg.to_config(self._tpot.fitted_pipeline_),
            indent=4,
        )
        self._log(f"Fitted pipeline:\n\n{pretty_fitted_pipeline}\n")
        if self.end_params.clear_population_file:
            (self.pop_path_).unlink(missing_ok=True)
        self._log("End of segment; Run complete\n")
