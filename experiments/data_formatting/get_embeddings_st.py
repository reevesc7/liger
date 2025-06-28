import pandas as pd
from liger.embedding.sentencetrans import STEmbedder
import constants as cnst
import get_prompts


def embeddings_st() -> pd.DataFrame:
    prompts = get_prompts.prompts()
    if cnst.EMBEDDINGS_ST_OP == cnst.Op.MAKE:
        embedder = STEmbedder(cnst.ST_EMBED_MODEL)
        return embedder.embed(prompts)
    return pd.read_csv(cnst.EMBEDDINGS_ST_FILE)


def main():
    embeddings_st().to_csv(cnst.EMBEDDINGS_ST_FILE, index=False)


if __name__ == "__main__":
    main()

