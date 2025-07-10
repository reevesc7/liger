import pandas as pd
from liger import smallville
import config as cfg


def prompts() -> pd.Series:
    if cfg.PROMPTS_OP == cfg.Op.MAKE:
        return smallville.get_logged_prompts(
            cfg.LOG_FILE,
            cfg.PROMPT_START,
            cfg.PROMPT_END,
            cfg.PROMPT_WHITELIST,
            cfg.PROMPT_BLACKLIST,
        )
    prompts = pd.read_csv(cfg.PROMPTS_FILE).squeeze(axis=1)
    if not isinstance(prompts, pd.Series):
        raise TypeError(f"prompts read from {prompts} is not a pandas.Series")
    return prompts


def main():
    prompts().to_csv(cfg.PROMPTS_FILE, index=False)


if __name__ == "__main__":
    main()

