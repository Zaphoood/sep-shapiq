"""Example test case for the Explainer class."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from shapiq import Explainer, InteractionValues
from sklearn.neighbors import KNeighborsClassifier

from shapiq_student import KNNExplainer

DATA_DIM = 2  # number of dimensions in the KNNExplainer's data parameter expects
LABELS_DIM = 1  # number of dimensions in the KNNExplainer's labels parameter expects


class TestExplainer:
    """Tests for the KNNExplainer class."""

    def test_is_explainer_class(self):
        """Test if KNNExplainer is a subclass of shapiq's Explainer."""
        assert issubclass(KNNExplainer, Explainer), "KNNExplainer should be subclass of Explainer."

    def test_init(self, knn_basic, data_train):
        """Test the initialization of KNNExplainer."""
        x_train, y_train = data_train
        assert isinstance(knn_basic, KNeighborsClassifier)
        assert isinstance(x_train, np.ndarray)
        assert len(x_train.shape) == DATA_DIM, "KNNExplainer data should be a 2D numpy array."
        assert isinstance(y_train, np.ndarray)
        assert len(y_train.shape) == LABELS_DIM, "KNNExplainer labels should be a 1D numpy array."
        explainer = KNNExplainer(model=knn_basic, data=x_train, labels=y_train)
        assert isinstance(explainer, KNNExplainer)
        assert isinstance(explainer, Explainer)
        assert explainer.max_order == 1, "We only expect Shapley value so this has to be 1."
        assert explainer.index == "SV", "Index should always be SV."

    def test_explain_function(self, knn_basic, data_train, x_explain):
        """Test the explain function of KNNExplainer works."""
        x_train, y_train = data_train
        n_samples = x_train.shape[0]
        explainer = KNNExplainer(model=knn_basic, data=x_train, labels=y_train)
        # Note: here we intentionally use the Explainer.explain function and not the KNNExplainer's
        # internal explain_function method to also ensure consistency in the interface.
        explanation = explainer.explain(x=x_explain)
        assert explanation is not None, "Explanation should not be None."
        assert isinstance(explanation, InteractionValues)
        assert explanation.index == "SV", "Index of the explanation should be SV."
        assert explanation.max_order == 1, "Max order should be 1 for KNNExplainer."
        assert explanation.n_players == n_samples, "Number of players should be n_samples."

    @pytest.mark.parametrize(
        ("model_name", "expected_mode"),
        [
            ("knn_basic", "normal"),
            ("knn_weighted", "weighted"),
            ("knn_radius", "threshold"),
        ],
    )
    def test_different_model_types(self, model_name, expected_mode, data_train, x_explain, request):
        """Test if KNNExplainer works with different model types."""
        x_train, y_train = data_train
        # get fixture by name
        model = request.getfixturevalue(model_name)
        n_samples = x_train.shape[0]
        explainer = KNNExplainer(model=model, data=x_train, labels=y_train)
        assert explainer.mode == expected_mode  # check if the KNNExplainer sets a mode correctly
        explanation = explainer.explain(x=x_explain)
        assert explanation is not None, "Explanation should not be None."
        assert isinstance(explanation, InteractionValues)
        assert explanation.index == "SV", "Index of the explanation should be SV."
        assert explanation.max_order == 1, "Max order should be 1 for KNNExplainer."
        assert explanation.n_players == n_samples, "Number of players should be n_samples."

    def test_class_index(self, knn_basic_multiclass, data_train_multiclass, x_explain_multiclass):
        """Test if KNNExplainer works with multiclass classification by specifying a class index."""
        x_train, y_train = data_train_multiclass

        # get the class with highest probability
        predicted_class = knn_basic_multiclass.predict(x_explain_multiclass)[0]
        proba = knn_basic_multiclass.predict_proba(x_explain_multiclass)[0][predicted_class]
        message = (
            f"KNN (Multiclass) model predicted class: {predicted_class} "
            f"with probability: {proba:.4f}"
        )
        logging.info(message)

        # make explanation for the predicted class
        explainer = KNNExplainer(
            model=knn_basic_multiclass, data=x_train, labels=y_train, class_index=predicted_class
        )
        explanation_pred_class = explainer.explain(x=x_explain_multiclass)
        explanation_pred_class_second = explainer.explain(x=x_explain_multiclass)

        # make explanation for different class and check if it is different
        other_class_index = 0
        explainer = KNNExplainer(
            model=knn_basic_multiclass,
            data=x_train,
            labels=y_train,
            class_index=other_class_index,
        )
        explanation_other_class = explainer.explain(x=x_explain_multiclass)

        # check that explanations are not the same
        assert explanation_pred_class != explanation_other_class
        assert explanation_pred_class_second != explanation_other_class
        assert explanation_pred_class == explanation_pred_class_second, (
            "Explanations for the same input and class should be equal."
        )
