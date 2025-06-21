from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from shapiq_student.explainer import knn as knn

df = load_iris()

# print(df)

X = df.data
y = df.target

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisieren der Modelle
probierModel = KNeighborsClassifier(n_neighbors=20)

# Trainieren der Modelle
probierModel.fit(X_train, y_train)

# Vorhersagen
y_pred_prob = probierModel.predict(X=X_test)
y_pred_probMitWahrsch = probierModel.predict_proba(X=X_test)

for i in range(5):
    nachbarn = probierModel.kneighbors(
        X=[X_test[i]], n_neighbors=len(X_train), return_distance=False
    )
    print(nachbarn)

    expl = knn.KNNClassifierExplainer(data=None, model=probierModel, class_index=y_test[i])

    result = expl.explain_function(X_test=X_test[i])

    print(result)
    sum_result = np.sum(result)
    print(sum_result)

# print("next")
# for j in range (0,10):
#     testoutput = [0,0,0]
#     for i in range(0,3):
#         knn_expl = knn.KNNClassifierExplainer(data=None, model=probierModel, class_index=i)
#         testoutput[i] = abs(sum(knn_expl.explain_function(X_test=X_test[j])))
#     print(sum(testoutput))

#     print(np.allclose(np.sum(testoutput), 1))
