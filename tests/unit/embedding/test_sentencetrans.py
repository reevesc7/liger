import pytest
import pandas as pd
from liger.embedding.sentencetrans import STEmbedder


@pytest.mark.torch
def test_st_embedder_embed() -> None:
    embedder = STEmbedder("all-mpnet-base-v2")
    embedding = embedder.embed("This is a test sentence.")
    assert isinstance(embedding, pd.Series)
    assert embedding.shape == (768,)
    embedding = embedder.embed(["This is a test sentence.", "This is another."])
    assert isinstance(embedding, pd.DataFrame)
    assert embedding.shape == (2, 768)
