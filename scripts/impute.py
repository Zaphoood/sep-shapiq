"""Tests using the GaussianImputer with an Explainer."""

from __future__ import annotations

from typing import cast

from pandas import DataFrame, Series
from shapiq import TabularExplainer
from shapiq.datasets import load_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapiq_student import GaussianImputer

X, y = cast(tuple[DataFrame, Series], load_california_housing())

X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.25,
    random_state=42,
)
n_features = X_train.shape[1]
model = RandomForestRegressor(
    max_depth=n_features,
    max_features=2 / 3,
    max_samples=2 / 3,
    random_state=42,
)
model.fit(X_train, y_train)


gaussian_imputer = GaussianImputer(model=model.predict, data=X_train, n_mc_samples=100)

explainer_gaussian = TabularExplainer(
    model=model, data=X_train, index="SII", max_order=2, imputer=gaussian_imputer
)

gaussian_values = explainer_gaussian.explain(X_test[0], budget=2**n_features)
print(gaussian_values)
