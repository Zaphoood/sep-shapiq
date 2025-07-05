"""Custom types used by KNN Explainers."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

KNNClassifierModel = KNeighborsClassifier | RadiusNeighborsClassifier
