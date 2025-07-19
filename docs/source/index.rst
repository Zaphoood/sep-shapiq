.. shapiq_student documentation master file, created by
   sphinx-quickstart on Sun Jun  1 15:12:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The ``shapiq_student`` Python package
=====================================

The ``shapiq_student`` package is an extension to the ``shapiq`` library for explaining machine learning models with Shapley interactions.
It adds the following functionalities:

1. **Explainers:** Easy-to-use explainers implementing efficient algorithms for explaining nearest neighbor models, including unweighted and weighted $k$-nearest neighbor classifiers, as well as threshold nearest neighbor classifiers
2. **Imputers:** Offers a *Gaussian* imputer for normally distributed features, plus a *Gaussian copula* variant that accommodates multivariate dependencies when marginals deviate from normality.
3. **Coalition Finding:** Implements an efficient, heuristic algorithm for finding maximal and minimal coalitions for a (simplified) game

To get a short overview of the library's features, have a look at the :ref:`quick start <quick-start>` below.
For a more in-depth tutorial, follow one of the :doc:`notebooks <notebooks>`. They provide a hands-on introduction to the usage of the library's modules while also explaining some of the theoretical concepts involved.
Details about the library's interfaces can be found in the :doc:`API reference <api>`.

.. warning::
    The ``Explainer``\ s implemented here rely on undocumented implementation details of ``scikit-learn`` for extracting training data from the model they explain.
    As a result, compatibility is not guaranteed across versions, and these implementations may break with future updates of ``scikit-learn``.
    The functionality has been tested and is confirmed to work only for ``scikit-learn==1.7.0``.

Contents
--------

.. toctree::
   :maxdepth: 2

   Notebooks <notebooks>

.. toctree::
   :maxdepth: 2

   api

   references

.. _quick-start:

Quick Start
-----------

Explainers
~~~~~~~~~~

Here's a short example showing how to use ``KNNExplainer`` to explain the prediction of a weighted ``KNeighborsClassifier``:

.. code-block:: python

    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from shapiq_student.explainer.knn import KNNExplainer, interaction_values_to_array
    >>>
    >>> X, y = make_classification(n_samples=40)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    >>> # Create a weighted classifier using `weights="distance"`
    >>> model = KNeighborsClassifier(n_neighbors=7, weights="distance")
    >>> model.fit(X_train, y_train)
    KNeighborsClassifier(n_neighbors=7, weights='distance')
    >>>
    >>> explainer = KNNExplainer(model, class_index=0) # Explain the prediction for class 0
    >>> type(explainer)  # Explainer for weighted KNN models is selected automatically
    <class 'shapiq_student.explainer.knn.weighted_knn.WeightedKNNExplainer'>
    >>> iv = explainer.explain(X_test[0])
    >>> sv = interaction_values_to_array(iv)  # Convert to array for easier handling
    >>> sv
    array([ 0.06971687,  0.03333333,  0.09914777, ... ])
    >>> sv.shape[0] == X_train.shape[0]  # Every training data point is assigned a Shapley value
    True

Imputers
~~~~~~~~

This example uses the ``GaussianImputer`` to explain the prediction of a random forest model trained on the California Housing dataset.

.. code-block:: python

    >>> from shapiq.datasets import load_california_housing
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> from shapiq import TabularExplainer
    >>> from shapiq_student import GaussianImputer
    >>>
    >>> X, y = load_california_housing()
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X.values,
    ...     y.values,
    ...     test_size=0.25,
    ...     random_state=42,
    ... )
    >>> model = RandomForestRegressor(
    ...     max_depth=X_train.shape[1], max_features=2 / 3, max_samples=2 / 3, random_state=42,
    ... )
    >>> model.fit(X_train, y_train)
    RandomForestRegressor(...)
    >>>
    >>> gaussian_imputer = GaussianImputer(model=model.predict, data=X_train, n_mc_samples=1000)
    >>> explainer = TabularExplainer(
    ...     model=model, data=X_train, index="SII", max_order=2, imputer=gaussian_imputer
    ... )
    >>> explainer.explain(X_test[0], budget=2**X_train.shape[1])
    InteractionValues(
        index=SII, max_order=2, min_order=0, estimated=False, estimation_budget=256,
        n_players=8, baseline_value=0.0,
        Top 10 interactions:
            (): 1.5283145711993955
            (4, 7): 0.3442686728461907
            (1, 6): 0.3318342718849081
            # ... more interactions
    )

*(The outputs in the examples above have been truncated for readability.)*
