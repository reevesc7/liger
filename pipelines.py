from os import makedirs, remove
from os.path import exists
from time import sleep
from sentence_transformers import SentenceTransformer
import pandas as pd
from .data_io import add_sentences, embed_dataframe
from .dataset import Dataset
from .training_testing import kfold_predict
from tpot import TPOTClassifier
from tpot.config import classifier_config_dict
from sklearn.model_selection import KFold
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score


def tpot_pipeline(training_data: str, output_directory: str, prompt_text: str, n_generations: int, pop_size: int, tpot_random_states: list[int], eval_random_states: list[int], no_trees: bool = False) -> None:

    # Import dataset
    TRANSFORMER_DIM = 768
    sentence_transformer = SentenceTransformer('all-mpnet-base-v2')
    dataframe = pd.read_csv(training_data)
    dataframe_alter = add_sentences(dataframe, prompt_text)
    dataset = embed_dataframe(model=sentence_transformer, data=dataframe_alter, feature_keys=dataframe.columns[:-1], score_key='rating', model_dim=TRANSFORMER_DIM)

    # Save prep
    training_data_name = training_data.rsplit('/')[-1].split('.')[0]
    output_directory = output_directory.strip('/') + '/' + training_data_name + '/'
    performance_file, ratings_file = save_prep(output_directory, dataframe["prompt"])

    # TPOT
    for tpot_random_state in tpot_random_states:
        print("\nRunning pipeline (n_gens = " + str(n_generations) + ", pop_size = " + str(pop_size) + ", trees_allowed = " + str(not no_trees) + ", random_state = " + str(tpot_random_state) + ")", flush=True)
        tpot = tpot_fit(dataset, n_generations, pop_size, tpot_random_state, no_trees)
        tpot.export(output_directory + "n_gens_" + str(n_generations) + "_popsize_" + str(pop_size) + "_tpotrs_" + str(tpot_random_state) + ".py")
        training_score = tpot.score(dataset.X, dataset.y)
        for eval_random_state in eval_random_states:
            kfold_predictions, kfold_r2 = tpot_test(tpot.fitted_pipeline_, dataset, eval_random_state)
            wait_for_free_csv(output_directory)
            with open(output_directory + "unsafe.txt", 'w') as _us:
                pass
            ratings = pd.read_csv(ratings_file)
            eval_name = "n_gens_" + str(n_generations) + "_pop_size_" + str(pop_size) + "_tpotrs_" + str(tpot_random_state) + "_evalrs_" + str(eval_random_state)
            ratings.insert(ratings.shape[1], eval_name, kfold_predictions, True)
            ratings.to_csv(ratings_file, index=False)
            performances = pd.read_csv(performance_file)
            performances.loc[performances.shape[0]] = pd.Series({'prompt_text': prompt_text, 
                                                                 'n_gens': n_generations, 
                                                                 'pop_size': pop_size, 
                                                                 'trees_allowed': not no_trees, 
                                                                 'tpot_random_state': tpot_random_state, 
                                                                 'eval_random_state': eval_random_state, 
                                                                 'training_score': training_score, 
                                                                 'n_splits': len(dataset.X), 
                                                                 'KFold_R2': kfold_r2
                                                                 })
            performances.to_csv(performance_file, index=False)
            remove(output_directory + "unsafe.txt")


def save_prep(output_directory: str, prompts: pd.Series) -> tuple[str, str]:
    performance_file = output_directory + "performance.csv"
    ratings_file = output_directory + "ratings.csv"
    if not exists(output_directory):
        makedirs(output_directory)
    if not exists(performance_file):
        pd.DataFrame(columns=['prompt_text', 'n_gens', 'pop_size', 'trees_allowed', 'tpot_random_state', 'eval_random_state', 'training_score', 'n_splits', 'KFold_R2']).to_csv(performance_file, index=False)
    if not exists(ratings_file):
        pd.DataFrame(index=[prompts]).to_csv(ratings_file)
    return performance_file, ratings_file


def tpot_fit(dataset: Dataset, n_generations: int, pop_size: int, tpot_random_state: int, no_trees: bool = False) -> TPOTClassifier:
    custom_config = None
    if no_trees:
        custom_config = {key: value for key, value in classifier_config_dict.items() if 'RandomForest' not in key and 'Tree' not in key}
    tpot = TPOTClassifier(config_dict=custom_config, generations=n_generations, population_size=pop_size, verbosity=1, random_state=tpot_random_state)
    tpot.fit(dataset.X, dataset.y)
    return tpot


def tpot_test(model, dataset: Dataset, eval_random_state: int) -> tuple[list[float], float]:
    set_param_recursive(model.steps, 'random_state', eval_random_state)
    kfold = KFold(n_splits=len(dataset.X), shuffle=True, random_state=eval_random_state)
    kfold_predictions = kfold_predict(model, kfold, dataset)
    kfold_r2 = r2_score(kfold_predictions, dataset.y)
    return kfold_predictions, kfold_r2


def wait_for_free_csv(directory: str) -> None:
    loop_count = 0
    while exists(directory + "unsafe.txt"):
        sleep(0.1)
        loop_count += 1
        if loop_count > 100:
            print("ERROR: Output csv possibly stuck marked as not safe for writing.")
