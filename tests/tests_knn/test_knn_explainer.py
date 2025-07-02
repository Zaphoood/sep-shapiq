"""Test the functionality of KNNClassifierExplainer."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from shapiq_student.explainer.knn import KNNClassifierExplainer


def test_output_length():
    """Tests wether the explain_function() returns an output in a valid format."""
    dataf = load_iris()
    X = dataf.data
    y = dataf.target

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probierModel = KNeighborsClassifier(n_neighbors=20)
    probierModel.fit(X_train, y_train)

    knn_expl = KNNClassifierExplainer(model=probierModel, class_index=y_test[5])

    testoutput = knn_expl.explain_function(X_test=X_test[5])

    assert (len(testoutput)) == (len(y_train))


def test_output_range():
    """Tests wether the calculated shapley values are in range ](1/n);1] with n = number of class indices."""
    dataf = load_iris()
    X = dataf.data
    y = dataf.target

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probierModel = KNeighborsClassifier(n_neighbors=20)
    probierModel.fit(X_train, y_train)

    outarray = np.zeros(len(y_test))

    for i in range(len(y_test)):
        knn_expl = KNNClassifierExplainer(model=probierModel, class_index=y_test[i])
        outarray[i] = np.sum(knn_expl.explain_function(X_test=X_test[i]))

    def between_all_and(arr, a, b):
        return np.all((arr > a) & (arr < b))

    assert between_all_and(outarray, 0.33333, 1.00001)


def test_sum_output():
    """Tests wether the returned shapley values sum up to 1 when calculated for all indices."""
    dataf = load_iris()
    X = dataf.data
    y = dataf.target

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    probierModel = KNeighborsClassifier(n_neighbors=20)
    probierModel.fit(X_train, y_train)

    for j in range(5):
        testoutput = [0, 0, 0]
        for i in range(3):
            knn_expl = KNNClassifierExplainer(model=probierModel, class_index=i)
            testoutput[i] = knn_expl.explain_function(X_test=X_test[j])

        assert np.allclose(np.sum(testoutput), 1)


# def test_compare_brute_force_explainer():
#     """Tests wether the output equals the output of brute-force-explanation"""
#     dataf = load_iris()
#     X = dataf.data
#     y = dataf.target

#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     X = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     probierModel = KNeighborsClassifier(n_neighbors=20)
#     probierModel.fit(X_train, y_train)

#     n = len(y_train)
#     permut = 0
#     for j in range(n-1):
#         x = math.comb(n-1,j)
#         permut = permut + x
#     comp_out = np.zeros(n)
#     ergsum = 0
#     for i in range(n):
#         trX = np.delete(X_train,[i])
#         trY = np.delete(y_train,[i])
#         count_amount_of_lengths = np.zeros(n-1)
#         for k in range(n-2):
#             X_combos = np.array(list(itertools.combinations(trX, k+1)))
#             y_combos = np.array(list(itertools.combinations(trY, k+1)))
#             modelk = KNeighborsClassifier(n_neighbors=20)
#             modelk.fit(X_combos, y_combos)
#             result_without = modelk.predict(X_test)
#             X_combos_plus = np.append(X_combos[0], trX[i])
#             y_combos_plus = np.append(y_combos[0], trY[i])
#             modelk_plus = KNeighborsClassifier(n_neighbors=20)
#             modelk_plus.fit(X_combos_plus, y_combos_plus)
#             result_with = modelk_plus.predict(X_test)
#             if (result_without[1] != result_with[1]):
#                 if (result_with[1] == y_test[1]):
#                     ergsum = ergsum + 1
#                 else:
#                     ergsum = ergsum - 1
#         comp_out[i] = (ergsum)/(permut)
