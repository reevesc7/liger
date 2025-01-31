from os import makedirs, remove
from os.path import exists
from time import sleep
import pandas as pd
from ..dataset import Dataset
from ..training_testing import kfold_predict
from tpot import TPOTClassifier, TPOTRegressor
from tpot.config import classifier_config_dict, regressor_config_dict
from sklearn.model_selection import KFold
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score


def tpot_pipeline(
    training_data: str,
    output_directory: str,
    regression: bool,
    n_generations: int,
    pop_size: int,
    tpot_random_states: list[int],
    eval_random_states: list[int],
    n_split_generations: int = None,
    no_trees: bool = False,
    no_xg: bool = False,
    cutoff_mins: int = None,
    n_splits: int = None,
) -> None:

    # Import dataset, use regression if float ratings
    dataframe = pd.read_csv(training_data)
    
    # TODO this won't work with multivalue prediction \/
    if (not regression) and any([isinstance(y, float) for y in dataframe['y']]):
        print("WARNING: Float data detected, using regression.")
        regression = True
    dataset = Dataset.from_df(dataframe)
    if n_splits == None:
        n_splits = dataset.X.shape[0]

    # Save prep
    training_data_name = training_data.rsplit('/')[-1].split('.')[0]
    output_directory = output_directory.strip('/') + '/' + training_data_name + '/'
    performance_file, ratings_file = save_prep(output_directory, dataframe['y'])

    # TPOT
    for tpot_random_state in tpot_random_states:
        print("\nRunning pipeline (n_gens = " + str(n_generations) + ", pop_size = " + str(pop_size) + ", trees_allowed = " + str(not no_trees) + ", random_state = " + str(tpot_random_state) + ")", flush=True)
        tpot = tpot_init(dataset, regression, n_generations, pop_size, tpot_random_state, no_trees, no_xg, cutoff_mins)
        tpot.fit(dataset.X, dataset.y)
        # tpot = tpot_fit(dataset, regression, n_generations, pop_size, tpot_random_state, no_trees)
        tpot.export(output_directory + "n_gens_" + str(n_generations) + "_popsize_" + str(pop_size) + "_tpotrs_" + str(tpot_random_state) + ".py")
        training_score = tpot.score(dataset.X, dataset.y)
        for eval_random_state in eval_random_states:
            kfold_predictions, kfold_r2 = tpot_test(tpot.fitted_pipeline_, dataset, eval_random_state, n_splits)
            wait_for_free_csv(output_directory)
            with open(output_directory + "unsafe.txt", 'w') as _us:
                pass
            ratings = pd.read_csv(ratings_file)
            eval_name = "n_gens_" + str(n_generations) + "_pop_size_" + str(pop_size) + "_tpotrs_" + str(tpot_random_state) + "_evalrs_" + str(eval_random_state)
            ratings.insert(ratings.shape[1], eval_name, list(kfold_predictions), True)
            ratings.to_csv(ratings_file, index=False)
            performances = pd.read_csv(performance_file)
            performances.loc[performances.shape[0]] = pd.Series(
                {
                    'regression': regression, 
                    'n_gens': n_generations, 
                    'pop_size': pop_size, 
                    'trees_allowed': not no_trees, 
                    'tpot_random_state': tpot_random_state, 
                    'eval_random_state': eval_random_state, 
                    'training_score': training_score, 
                    'n_splits': n_splits, 
                    'KFold_R2': kfold_r2
                }
            )
            performances.to_csv(performance_file, index=False)
            remove(output_directory + "unsafe.txt")


def save_prep(output_directory: str, prompts: pd.Series) -> tuple[str, str]:
    performance_file = output_directory + "performance.csv"
    ratings_file = output_directory + "ratings.csv"
    if not exists(output_directory):
        makedirs(output_directory)
    if not exists(performance_file):
        pd.DataFrame(columns=[
            'regression',
            'n_gens',
            'pop_size',
            'trees_allowed',
            'tpot_random_state',
            'eval_random_state',
            'training_score',
            'n_splits',
            'KFold_R2'
        ]).to_csv(performance_file, index=False)
    if not exists(ratings_file):
        pd.DataFrame(index=[prompts]).to_csv(ratings_file)
    return performance_file, ratings_file


def tpot_init(regression: bool, n_generations: int, pop_size: int, tpot_random_state, no_trees: bool, no_xg: bool, cutoff_mins: int):
    if regression:
        custom_config = regressor_config_dict
    else:
        custom_config = classifier_config_dict
    if no_trees:
        custom_config = {key: value for key, value in custom_config.items() if 'RandomForest' not in key and 'Tree' not in key}
    if no_xg:
        custom_config = {key: value for key, value in custom_config.items() if 'XG' not in key}
    if regression:
        tpot = TPOTRegressor(
            config_dict=custom_config,
            generations=n_generations,
            population_size=pop_size,
            verbosity=2,
            random_state=tpot_random_state,
            max_time_mins=cutoff_mins,
        )
    else:
        tpot = TPOTClassifier(
            config_dict=custom_config,
            generations=n_generations,
            population_size=pop_size,
            verbosity=2,
            random_state=tpot_random_state,
            max_time_mins=cutoff_mins,
        )
    return tpot


def tpot_test(model, dataset: Dataset, eval_random_state: int, n_splits: int) -> tuple[list[float], float]:
    set_param_recursive(model.steps, 'random_state', eval_random_state)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=eval_random_state)
    kfold_predictions = kfold_predict(model, kfold, dataset)
    kfold_r2 = r2_score(kfold_predictions, dataset.y)
    return kfold_predictions, kfold_r2


def wait_for_free_csv(directory: str) -> None:
    loop_count = 0
    while exists(directory + "unsafe.txt"):
        sleep(0.1)
        loop_count += 1
        if loop_count > 100:
            print("WARNING: Output csv possibly stuck marked as not safe for writing.")
            return
