"""Test the functionality of KNNClassifierExplainer."""

from __future__ import annotations

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from shapiq_student.explainer.knn import KNNClassifierExplainer


def test_explain_function():
    """Tests wether the explain_function() returns a output in a valid format."""
    dataf = load_iris()
    X = dataf.data
    y = dataf.target

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probierModel = KNeighborsClassifier(n_neighbors=20)
    probierModel.fit(X_train, y_train)

    knn_expl = KNNClassifierExplainer(data=None, model=probierModel, class_index=y_test[5])

    testoutput = knn_expl.explain_function(X_test=X_test[5])

    assert (len(testoutput)) == (len(y_train))
