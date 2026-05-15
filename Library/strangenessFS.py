"""Backward-compatible strangeness utilities."""

from maxrel_minred_minstra._strangeness import (
    SupportVectorOneVsRestClassifier as CustomOneVsRestClassifier,
    strangeness_scores,
)


def beta_measures(X, Y, classes_, split_size=0.5, lambda_=0.5, kernel="linear") -> list:
    return strangeness_scores(
        X,
        Y,
        classes=classes_,
        split_size=split_size,
        lambda_=lambda_,
        kernel=kernel,
    ).tolist()
