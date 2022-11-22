"""Provide scikit-learn estimator for kernel ridge classification."""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score


class KernelRidgeClassifier(KernelRidge):
    """Scikit-learn estimator for kernel ridge classification.

    Fits a KernelRidge Regressor to the binary labels 0 and 1 and
    then uses 0.5 as a threshold to binarise the predictions.
    """

    def __init__(
        self,
        alpha=1.0,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
    ):
        """Initialise class."""
        super().__init__(
            alpha=alpha,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            kernel_params=kernel_params,
        )

    def fit(self, X, y):
        """Implement fit."""
        for val in np.unique(y):
            assert val in [0, 1], "Target values should be 0 or 1!"

        super().fit(X, y)
        return self

    def predict(self, X):
        """Implement predict."""
        return np.array([0 if x < 0.5 else 1 for x in super().predict(X)])

    def score(self, X, y, sample_weight=None):
        """Implement score."""
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
