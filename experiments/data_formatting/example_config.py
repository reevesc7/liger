from enum import Enum
from pathlib import Path


class Op(Enum):
    SKIP = 0
    MAKE = 1
    RETR = 2


# OPERATIONS: whether to skip, make, or retrieve each data type
PROMPTS_OP         = Op.MAKE
RESPONSES_OP       = Op.MAKE
FUNCTIONALS_OP     = Op.MAKE
EMBEDDINGS_AI_OP   = Op.MAKE
EMBEDDINGS_ST_OP   = Op.MAKE


# FILES: file paths to retrieve data from and write data to
DIRECTORY          = Path("smallville_846")
LOG_FILE           = DIRECTORY / "simulation_test_013_2024-10-24.txt"
PROMPTS_FILE       = DIRECTORY / "prompts.csv"
RESPONSES_FILE     = DIRECTORY / "responses.csv"
FUNCTIONALS_FILE   = DIRECTORY / "functionals.csv"
EMBEDDINGS_AI_FILE = DIRECTORY / "embeddings_ai.csv"
EMBEDDINGS_ST_FILE = DIRECTORY / "embeddings_st.csv"
DATASET_FILE       = DIRECTORY / "smallville_846.csv"


# SAVE INTERMEDIATES: whether to save prompts.csv, etc. when running format_dataset.py
SAVE_INTERMEDIATES = True

# PROMPT RETRIEVAL PATTERN: only prompts containing this substring will be retrieved
PROMPT_PATTERN     = "rate the likely poignancy"

# SURVEYING: parameters for controlling response generation
AI_SURVEY_MODEL    = "gpt-3.5-turbo"
RESPONSE_SEED      = "{\"output\": \""
ALLOWED_TOKENS     = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

# OPENAI EMBEDDING: OpenAI embedding model to use
AI_EMBED_MODEL     = "text-embedding-3-small"

# SENTENCE TRANSFORMERS EMBEDDING: sentencetransformers model to use
ST_EMBED_MODEL     = "all-mpnet-base-v2"

