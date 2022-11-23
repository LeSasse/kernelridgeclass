"""Provide scikit-learn estimator for kernel ridge classification."""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score


class KernelRidgeClassifier(KernelRidge):
    """Scikit-learn estimator for kernel ridge classification.

    Fits a KernelRidge Regressor to the binary labels 0 and 1 and
    then uses 0.5 as a threshold to binarise the predictions.
    """

    def fit(self, X, y):
        """Implement fit."""
        for val in np.unique(y):
            assert val in [0, 1], "Target values should be 0 or 1!"

        super().fit(X, y)
        return self

    def predict(self, X):
        """Implement predict."""
        return np.where(super().predict(X) > 0.5, 0, 1)

    def score(self, X, y, sample_weight=None):
        """Implement score."""
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
