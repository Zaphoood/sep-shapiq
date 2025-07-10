"""Stresstest for Explainers.

Stresstest for the Explainer functions.
"""

from __future__ import annotations

import time

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student.explainer.knn import (
    KNNClassifierExplainer,
)

start_time = time.time()

higgs = datasets.fetch_openml(data_id=45570, as_frame=False)

load_timer = time.time()

X_train, X_test, y_train, _ = train_test_split(
    higgs.data, higgs.target, test_size=0.0001, random_state=42
)

preprocess_timer = time.time()

knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X_train, y_train)

fitting_timer = time.time()

knn_expl = KNNClassifierExplainer(model=knn_model, class_index=1)
knn_output = knn_expl.explain_function(x=X_test[0])

end_timer = time.time()


print(f"Größe Trainingsdatensatz: {len(y_train)}")  # noqa: T201
print(f"Dauer Preprocessing:      {preprocess_timer - load_timer}")  # noqa: T201
print(f"Dauer Modelerstellung:    {fitting_timer - preprocess_timer}")  # noqa: T201
print(f"Dauer Explaination:       {end_timer - fitting_timer}")  # noqa: T201
print(f"Gesamtdauer:              {end_timer - start_time}")  # noqa: T201

"""
 Größe Trainingsdatensatz: 10998900
 Dauer Preprocessing:      3.939413547515869
 Dauer Modelerstellung:    6.494971036911011
 Dauer Explaination:       8.916141986846924
 Gesamtdauer:              80.74211525917053
"""
