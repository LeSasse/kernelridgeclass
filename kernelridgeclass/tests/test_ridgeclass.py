"""Provide test for fcmodels."""

import numpy as np
import pandas as pd
from julearn import run_cross_validation
from sklearn.datasets import make_classification

from kernelridgeclass import KernelRidgeClassifier


def test_classification():
    """Provide test for kernel ridge classifier."""
    X, y = make_classification(random_state=0)
    clsf = KernelRidgeClassifier()
    clsf.fit(X, y)

    X_new, y_new = make_classification(random_state=0)
    clsf.score(X_new, y_new)


def test_classifier_with_julearn():
    """Test classifier as a drop-in model for julearn."""

    X, y = make_classification()

    # prepare data for julearn
    df = pd.DataFrame(X)
    features = [f"Feature {i}" for i, _ in enumerate(df.columns)]
    df.columns = features
    df["Target"] = y
    df["Confound"] = np.random.rand(df.shape[0])

    # prepare searcher
    clsf = KernelRidgeClassifier()

    # run julearn
    scores, model = run_cross_validation(
        data=df,
        X=features,
        y="Target",
        confounds="Confound",
        preprocess_X="remove_confound",
        model=clsf,
        problem_type="binary_classification",
        return_estimator="all",
    )
