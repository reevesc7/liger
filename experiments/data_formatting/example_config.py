from enum import Enum
from pathlib import Path


class Op(Enum):
    SKIP = 0
    MAKE = 1
    RETR = 2


# OPERATIONS: whether to skip, make, or retrieve each data type
PROMPTS_OP       = Op.MAKE
RESPONSES_OP     = Op.MAKE
FUNCTIONALS_OP   = Op.MAKE
EMBEDDINGS_AI_OP = Op.MAKE
EMBEDDINGS_ST_OP = Op.MAKE


# FILES: file paths to retrieve data from and write data to
DIRECTORY          = Path("smallville_846"); DIRECTORY.mkdir(parents=True, exist_ok=True)
LOG_FILE           = DIRECTORY / "simulation_prompts.txt"
PROMPTS_FILE       = DIRECTORY / "prompts.csv"
RESPONSES_FILE     = DIRECTORY / "responses.csv"
FUNCTIONALS_FILE   = DIRECTORY / "functionals.csv"
EMBEDDINGS_AI_FILE = DIRECTORY / "embeddings_ai.csv"
EMBEDDINGS_ST_FILE = DIRECTORY / "embeddings_st.csv"
DATASET_FILE       = DIRECTORY / "smallville_846.csv"


# SAVE INTERMEDIATES: whether to save prompts.csv, etc. when running format_dataset.py
SAVE_INTERMEDIATES = True

# PROMPTS: the substrings immediately preceding, following, and within desired prompts
#   Use line offsets to control which lines are included in the prompts.
#   (E.g., 0 includes that line, 1 includes 1 line after, -1 includes the line before).
PROMPT_START = {
    "Here is a brief description": -1,
}
PROMPT_END = {
    "\"output\"": 0,
}
PROMPT_PATTERN = "rate the likely poignancy"

# SURVEYING: parameters for controlling response generation
AI_SURVEY_MODEL = "gpt-3.5-turbo"
RESPONSE_SEED   = "{\"output\": \""
ALLOWED_TOKENS  = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}

# OPENAI EMBEDDING: OpenAI embedding model to use
AI_EMBED_MODEL = "text-embedding-3-small"

# SENTENCE TRANSFORMERS EMBEDDING: sentencetransformers model to use
ST_EMBED_MODEL = "all-mpnet-base-v2"

