# liger - Helper functions for the Likert General Regressor project
# Copyright (C) 2024  Chris Reeves
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


from typing import Any, Callable, Self
from sklearn.base import BaseEstimator, TransformerMixin, Tags
from sklearn.compose import TransformedTargetRegressor


REG_KEY = "regressor_"
TRANS_KEY = "transformer_"


def _maybe_instance(factory: Callable | None, kwargs: dict[str, Any]) -> Any:
    if factory is None:
        return None
    return factory(**kwargs)


def _filter_kwargs(kwargs: dict[str, Any], filter: str) -> dict[str, Any]:
    return {
        key.removeprefix(filter): value
        for key, value in kwargs.items()
        if key.startswith(filter)
    }


def flat_init_transformed_target_regressor(
    regressor_factory: Callable | None = None,
    transformer_factory: Callable | None = None,
    func: Callable | None = None,
    inverse_func: Callable | None = None,
    check_inverse: bool = True,
    **kwargs: Any,
) -> TransformedTargetRegressor:
    """Construct a `sklearn.compose.TransformedTargetRegressor` and internal
    regressor and transformer.
    #
    Parameters
    ----------
    `regressor_factory` : `Callable`, optional
        Constructor for a regressor.
    `transformer_factory` : `Callable`, optional
        Constructor for a transformer. Cannot be set at the same time as
        `func` and `inverse_func`. See `TransformedTargetRegressor`.
    `func` : `Callable`, optional
        Function used to tranform `y`. Cannot be set at the same time as
        `transformer_factory`. If set, `inverse_func` must also be set.
        See `TransformedTargetRegressor`.
    `inverse_func` : `Callable`, optional
        Function used to inverse tranform `y`. Cannot be set at the same time as
        `transformer_factory`. If set, `func` must also be set.
        See `TransformedTargetRegressor`.
    `check_inverse` : bool, default=`True`
        Whether to check if tranformation followed by inverse transformation returns
        the original `y`. See `TransformedTargetRegressor`.
    `kwargs` : `Any`
        Keyword arguments to pass to regressor and transformer factories.
        Prefix keys with `"regressor_"` and `"transformer_"` to pass arguments for
        regressor and transformer arguments, respectively.
    #
    Returns
    -------
    `regressor` : `sklearn.compose.TransformedTargetRegressor`
        Initialized regressor.
    """
    return TransformedTargetRegressor(
        regressor=_maybe_instance(
            regressor_factory,
            _filter_kwargs(kwargs, REG_KEY),
        ),
        transformer=_maybe_instance(
            transformer_factory,
            _filter_kwargs(kwargs, TRANS_KEY),
        ),
        func=func,
        inverse_func=inverse_func,
        check_inverse=check_inverse,
    )


class LgPassthrough(TransformerMixin, BaseEstimator):
    """Patched version of TPOT's Passthrough class.
    #
    All code has been copied from the `tpot/tpot/builtin_modules/passthrough.py`
    file of the `EpistasisLab/tpot` repository, with minor tweaks.
    #
    A transformer that does nothing. It just passes the input array as is.
    """

    def fit(self, _X=None, _y=None) -> Self:
        """Nothing to fit, returns self.
        """
        return self

    def transform(self, X: Any) -> Any:
        """Returns the input array as-is.
        """
        return X

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags
