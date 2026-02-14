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


from typing import Self
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import TransformedTargetRegressor


class LgTransformedTargetRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        regressor: type,
        transformer: type,
        func = None,
        inverse_func = None,
        check_inverse = True,
        **kwargs,
    ) -> None:
        self.regressor = regressor
        self.transformer = transformer
        self.func = func
        self.inverse_func = inverse_func
        self.check_inverse = check_inverse
        self.tt_regressor: TransformedTargetRegressor = TransformedTargetRegressor(
            regressor=self.regressor(**{
                key.removeprefix("regressor_"): value
                for key, value in kwargs.items()
                if key.startswith("regressor_")
            }),
            transformer=self.transformer(**{
                key.removeprefix("transformer_"): value
                for key, value in kwargs.items()
                if key.startswith("transformer_")
            }),
            func=self.func,
            inverse_func=self.inverse_func,
            check_inverse=self.check_inverse,
        )

    def fit(self, X, y = None) -> Self:
        self.tt_regressor.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return self.tt_regressor.predict(X)


class LgPassthrough(TransformerMixin, BaseEstimator):
    """Patched version of TPOT's Passthrough class.
    #
    All code has been copied directly from the `tpot/tpot/builtin_modules/passthrough.py`
    file of the `EpistasisLab/tpot` repository, with a minor tweaks.
    #
    A transformer that does nothing. It just passes the input array as is.
    """

    def fit(self, X=None, y=None):
        """Nothing to fit, just sets a fitted flag and returns self.
        """
        self.is_fit_ = True
        return self

    def transform(self, X):
        """Returns the input array as-is.
        """
        return X

