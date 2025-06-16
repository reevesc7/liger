import pandas as pd


def response_strip(responses: pd.Series) -> pd.Series:
    """Convert full Smallville poignancy responses to ints.

    Parameters
    ----------
    `responses` : `pandas.Series`
        The "response" column returned by `OpenAISurveyor. generate_response(),
        .generate_responses()`, `.survey()`. Elements are either `str`
        or `list[str]`, depending on number of replicates.

    Returns
    -------
    `stripped_responses` : `pandas.Series`
        The "response" column, with each element formatted as `int` if inputs
        were `str` or `list[int]` is inputs were `list[str]`.
    """
    if isinstance(responses[0], str):
        stripped = [
            int(rating.split(":")[-1].strip("}{\"\n\" "))
            for rating in responses
        ]
    else:
        stripped = [
            [int(rating.split(":")[-1].strip("}{\"\n\" ")) for rating in ratings]
            for ratings in responses
        ]
    return pd.Series(data=stripped, name=responses.name)

