# from __future__ import annotations
from __future__ import annotations

import itertools
import math

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

import shapiq_student
import shapiq_student.explainer
import shapiq_student.explainer.knn
import shapiq_student.explainer.knn.knn_explainer

df = load_iris()

# print(df)

X = df.data
y = df.target

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=41)

# Initialisieren der Modelle
probierModel = KNeighborsClassifier(n_neighbors=8)

# Trainieren der Modelle
probierModel.fit(X_train, y_train)

# # Vorhersagen
# y_pred_prob = probierModel.predict(X=X_test)
# y_pred_probMitWahrsch = probierModel.predict_proba(X=X_test)

# for i in range(5):
#     nachbarn = probierModel.kneighbors(
#         X=[X_test[i]], n_neighbors=len(X_train), return_distance=False
#     )
#     print(nachbarn)

#     expl = knn.KNNClassifierExplainer(data=None, model=probierModel, class_index=y_test[i])

#     result = expl.explain_function(X_test=X_test[i])

#     print(result)
#     sum_result = np.sum(result)
#     print(sum_result)

# print("next")
# for j in range (0,10):
#     testoutput = [0,0,0]
#     for i in range(0,3):
#         knn_expl = knn.KNNClassifierExplainer(data=None, model=probierModel, class_index=i)
#         testoutput[i] = abs(sum(knn_expl.explain_function(X_test=X_test[j])))
#     print(sum(testoutput))

#     print(np.allclose(np.sum(testoutput), 1))

# n = 6
# permut = 0
# for j in range(n-1):
#         x = math.comb(n-1,j)
#         permut = permut + x

# combis = np.array(list(itertools.combinations(X_test,3)))
# #npcombos = np.array(combis)

# print (combis)
#################################################################################################

n = len(y_train)
permut = 0
for j in range(n - 1):
    x = math.comb(n - 1, j)
    permut = permut + x
comp_out = np.zeros(n)
ergsum = 0
for i in range(n):
    #    if y_train[i] == y_test[3]:
    trX = np.delete(X_train, i, 0)
    trY = np.delete(y_train, i, 0)

    print(len(trX))
    print(len(trY))

    count_amount_of_lengths = np.zeros(n - 1)
    for k in range(n - 2):
        X_combos = np.array(list(itertools.combinations(trX, k + 1)))
        y_combos = np.array(list(itertools.combinations(trY, k + 1)))

        #            print(X_combos)
        #            print(len(X_combos[k]))
        print()
        print(len(y_combos[k]))
        print(i)
        print(k)

        n_neigh = min(8, len(y_combos[k]))
        n_neigh2 = min(8, len(y_combos[k]) + 1)

        for l in range(len(y_combos)):
            modelk = KNeighborsClassifier(n_neighbors=n_neigh)
            modelk.fit(X_combos[l], y_combos[l])
            result_without = modelk.predict([X_test[3]])

            X_combos_plus = np.append(X_combos[l], [X_train[i]], axis=0)
            y_combos_plus = np.append(y_combos[l], [y_train[i]], axis=0)
            modelk_plus = KNeighborsClassifier(n_neighbors=n_neigh2)
            modelk_plus.fit(X_combos_plus, y_combos_plus)
            result_with = modelk_plus.predict([X_test[3]])

            if np.allclose(result_without, result_with):
                ()
            elif np.allclose(result_with, [y_test[3]]):
                ergsum = ergsum + (1)
            else:
                ergsum = ergsum - (1)
            print(i)
            print()
            print(result_without)
            print(result_with)
            print([y_test[3]])
            print(np.allclose(result_with, [y_test[3]]))
            print(y_train[i])
            print()
    comp_out[i] = (ergsum) / (permut)
    ergsum = 0

print(comp_out)

expl = shapiq_student.explainer.knn.knn_explainer.KNNClassifierExplainer(
    model=probierModel, class_index=y_test[3]
)
shapiq_out = expl.explain_function(X_test=X_test[3])

print(shapiq_out)

print(y_test[3])
