.. shapiq_student documentation master file, created by
   sphinx-quickstart on Sun Jun  1 15:12:07 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The ``shapiq_student`` Python package
=====================================

The ``shapiq_student`` package is an extension to the the ``shapiq`` library for approximating Shapley interactions and explaining machine learning models.
It adds the following functionalities:

1. **Explainers:** Easy-to-use explainers implementing efficient algorithms for explaining nearest neighbor models, including unweighted and weighted $k$-nearest neighbor classifiers, as well as threshold nearest neighbor classifiers.
2. **Imputers:** Provides a *Gaussian* imputer for background data with normally distributed features, and its extension, the *Gaussian copula* imputer, suitable for arbitrary, non-normally distributed data
3. **Coalition Finding:** Implements an efficient, heuristic algorithm for finding maximal and minimal coalitions for a (simplified) game

The best way to get started is by following one of the :doc:`notebooks <notebooks>`, which provide a hands-on introduction to the usage of the library's modules while also explaining some of the theoretical concepts involved. Furthermore, an :doc:`API reference <api>` is provided.

Contents
~~~~~~~~

.. toctree::
   :maxdepth: 2

   Notebooks <notebooks>

.. toctree::
   :maxdepth: 2

   api

   citations
