import pandas as pd
from liger import smallville
import constants as cnst


def prompts() -> pd.Series:
    if cnst.PROMPTS_OP == cnst.Op.MAKE:
        return smallville.get_logged_prompts(cnst.LOG_FILE, cnst.PROMPT_PATTERN)
    prompts = pd.read_csv(cnst.PROMPTS_FILE).squeeze(axis=1)
    if not isinstance(prompts, pd.Series):
        raise TypeError(f"prompts read from {prompts} is not a pandas.Series")
    return prompts


def main():
    prompts().to_csv(cnst.PROMPTS_FILE, index=False)


if __name__ == "__main__":
    main()

