import pandas as pd
from liger.surveying.openai import OpenAISurveyor
import constants as cnst
import get_responses


def functionals() -> pd.DataFrame:
    responses = get_responses.responses()
    if cnst.FUNCTIONALS_OP == cnst.Op.MAKE:
        return OpenAISurveyor.functionals(responses)
    return pd.read_csv(cnst.FUNCTIONALS_FILE)


def main():
    functionals().to_csv(cnst.RESPONSES_FILE, index=False)


if __name__ == "__main__":
    main()

