import pandas as pd
from liger.surveying.openai import OpenAISurveyor
import config as cfg
import get_responses


def functionals() -> pd.DataFrame:
    if cfg.FUNCTIONALS_OP != cfg.Op.MAKE:
        return pd.read_csv(cfg.FUNCTIONALS_FILE)
    responses = get_responses.responses()
    return OpenAISurveyor.functionals(responses)


def main():
    functionals().to_csv(cfg.FUNCTIONALS_FILE, index=False)


if __name__ == "__main__":
    main()

