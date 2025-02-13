from pathlib import Path
from os.path import isfile
from openai import OpenAI
import numpy as np
import pandas as pd
from ..embedding import BaseEmbedder
from ... import Dataset


OPENAI_KEYFILE = "openai.key"


class LLMEmbedder(BaseEmbedder):
    def __init__(self, model_str: str):
        if not isfile(Path(OPENAI_KEYFILE)):
            raise FileNotFoundError(f"{OPENAI_KEYFILE} is not on $PATH")
        self.client = OpenAI(api_key=Path(OPENAI_KEYFILE).read_text().strip())
        self.model = model_str


    # Generates a dataset from the specified rows of a DataFrame.
    def embed_dataframe(self, data: pd.DataFrame, feature_keys: pd.Index, score_key: str) -> Dataset:
        n_entries = data.shape[0]
        n_features = len(feature_keys)

        # Hate how hacky this is
        feature_vectors = None
        for feature_index, feature_key in enumerate(feature_keys):
            for entry in range(n_entries):
                embedding = self.embed(str(data[feature_key][entry]))
                if feature_vectors is None:
                    feature_vectors = np.zeros((n_features, n_entries, len(embedding)))
                feature_vectors[feature_index][entry] = embedding
                print(feature_key, entry, "Finished", flush=True)
        feature_vectors = np.array(feature_vectors)
        dataset = Dataset(n_entries, n_features*feature_vectors.shape[2])
        dataset.X = feature_vectors.transpose((1, 0, 2)).reshape((n_entries, -1))
        dataset.y = np.array(data[score_key])
        return dataset


    def embed(self, input: str | None) -> list[float]:
        if input == None:
            input = ""
        input = input.replace("\n", " ")
        return self.client.embeddings.create(input=input, model=self.model).data[0].embedding

