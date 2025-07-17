"""Fixtures for the grading tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

if TYPE_CHECKING:
    import numpy as np

RANDOM_SEED: int = 42
"""Random seed for all randomness in the unit tests."""

N_SAMPLES: int = 1000


@pytest.fixture(scope="session")
def data() -> tuple[np.ndarray, np.ndarray]:
    """Fixture for a medium-sized dataset."""
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=20, n_informative=18, random_state=RANDOM_SEED
    )
    return X, y


@pytest.fixture(scope="session")
def data_split(data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture for a training dataset."""
    X, y = data
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


@pytest.fixture(scope="session")
def x_explain(data_split):
    """Fixture for a single data point to explain."""
    _, x_test, _, _ = data_split
    return x_test[0].reshape(1, -1)


@pytest.fixture(scope="session")
def data_train(data_split) -> tuple[np.ndarray, np.ndarray]:
    """Fixture for the training data."""
    x_train, _, y_train, _ = data_split
    return x_train, y_train


@pytest.fixture(scope="session")
def data_test(data_split) -> tuple[np.ndarray, np.ndarray]:
    """Fixture for the test data."""
    _, x_test, _, y_test = data_split
    return x_test, y_test


@pytest.fixture(scope="session")
def data_multiclass() -> tuple[np.ndarray, np.ndarray]:
    """Fixture for a medium-sized multiclass dataset."""
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=20, n_classes=5, n_informative=18, random_state=RANDOM_SEED
    )
    return X, y


@pytest.fixture(scope="session")
def data_multiclass_split(data_multiclass) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixture for a training dataset with a multiclass dataset."""
    X, y = data_multiclass
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


@pytest.fixture(scope="session")
def data_train_multiclass(data_multiclass_split) -> tuple[np.ndarray, np.ndarray]:
    """Fixture for the training data in a multiclass dataset."""
    x_train, _, y_train, _ = data_multiclass_split
    return x_train, y_train


@pytest.fixture(scope="session")
def data_test_multiclass(data_multiclass_split) -> tuple[np.ndarray, np.ndarray]:
    """Fixture for the test data in a multiclass dataset."""
    _, x_test, _, y_test = data_multiclass_split
    return x_test, y_test


@pytest.fixture(scope="session")
def x_explain_multiclass(data_multiclass_split) -> np.ndarray:
    """Fixture for a single data point to explain in a multiclass dataset."""
    _, x_test, _, _ = data_multiclass_split
    return x_test[0].reshape(1, -1)


@pytest.fixture
def knn_basic(data_train, data_test, x_explain) -> KNeighborsClassifier:
    """Fixture for a KNeighborsClassifier with 5 neighbors."""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(*data_train)
    score = knn.score(*data_test)
    message = f"KNN (Basic) model fitted with score: {score:.4f}"
    logging.info(message)
    prediction = knn.predict_proba(x_explain)
    message = f"KNN (Basic) prediction: {prediction}"
    logging.info(message)
    return knn


@pytest.fixture
def knn_weighted(data_train, data_test, x_explain) -> KNeighborsClassifier:
    """Fixture for a KNeighborsClassifier with 5 neighbors and weighted voting."""
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(*data_train)
    score = knn.score(*data_test)
    message = f"KNN (Weighted) model fitted with score: {score:.4f}"
    logging.info(message)
    prediction = knn.predict_proba(x_explain)
    message = f"KNN (Weighted) prediction: {prediction}"
    logging.info(message)
    return knn


@pytest.fixture
def knn_radius(data_train, data_test, x_explain) -> RadiusNeighborsClassifier:
    """Fixture for a KNeighborsClassifier with radius neighbors."""
    knn = RadiusNeighborsClassifier(radius=15)
    knn.fit(*data_train)
    score = knn.score(*data_test)
    message = f"KNN (Radius) model fitted with score: {score:.4f}"
    logging.info(message)
    prediction = knn.predict_proba(x_explain)
    message = f"KNN (Radius) prediction: {prediction}"
    logging.info(message)
    return knn


@pytest.fixture
def knn_basic_multiclass(
    data_train_multiclass, data_test_multiclass, x_explain_multiclass
) -> KNeighborsClassifier:
    """Fixture for a KNeighborsClassifier with 5 neighbors for multiclass classification."""
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(*data_train_multiclass)
    score = knn.score(*data_test_multiclass)
    message = f"KNN (Basic Multiclass) model fitted with score: {score:.4f}"
    logging.info(message)
    prediction = knn.predict_proba(x_explain_multiclass)
    message = f"KNN (Basic Multiclass) prediction: {prediction}"
    logging.info(message)
    return knn
