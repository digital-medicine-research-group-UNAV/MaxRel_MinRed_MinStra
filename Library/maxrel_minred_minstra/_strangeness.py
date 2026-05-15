from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class SupportVectorOneVsRestClassifier(OneVsRestClassifier):
    """One-vs-rest SVC wrapper exposing support vectors and dual coefficients."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.dual_coefs_ = []
        self.support_vectors_ = []
        self.estimators_ = []

        for class_label in self.classes_:
            estimator = clone(self.estimator)
            y_binary = np.where(y == class_label, 1, -1)
            estimator.fit(X, y_binary)
            self.dual_coefs_.append(estimator.dual_coef_)
            self.support_vectors_.append(estimator.support_vectors_)
            self.estimators_.append(estimator)

        return self


def strangeness_scores(
    X,
    y,
    classes,
    split_size: float = 0.5,
    lambda_: float = 0.5,
    kernel: str = "linear",
    random_state: int | None = None,
) -> np.ndarray:
    """Compute the minimal-strangeness feature scores used by mRMR_MS."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    classes = np.asarray(classes)

    if X.shape[1] == 0:
        return np.asarray([], dtype=float)
    if len(classes) < 2:
        raise ValueError("classes must contain at least two class labels.")

    X_train, X_cal, y_train, y_cal = train_test_split(
        X,
        y,
        test_size=split_size,
        stratify=y,
        random_state=random_state,
    )

    n_features = X_train.shape[1]
    lambda_p = (1.0 - lambda_) / (len(classes) - 1)

    if len(classes) == 2:
        positive_class = classes[-1]
        y_train_bin = np.where(y_train == positive_class, 1, -1)
        y_cal_bin = np.where(y_cal == positive_class, 1, -1)

        model = SVC(
            kernel=kernel,
            degree=3,
            tol=1e-3,
            probability=True,
            cache_size=900,
            C=1.0,
            max_iter=900000,
        )
        model.fit(X_train, y_train_bin)
        weights = np.dot(model.dual_coef_, model.support_vectors_).reshape(-1)

        beta = []
        for feature_idx in range(n_features):
            values = weights[feature_idx] * y_cal_bin * X_cal[:, feature_idx]
            beta.append(float(-np.sum(values)))
        return np.asarray(beta, dtype=float)

    class_to_index = {class_label: idx for idx, class_label in enumerate(classes)}
    y_cal_idx = np.asarray([class_to_index[label] for label in y_cal], dtype=int)

    model = SupportVectorOneVsRestClassifier(
        SVC(
            kernel=kernel,
            degree=3,
            tol=1e-3,
            probability=True,
            cache_size=900,
            C=1.0,
            max_iter=900000,
        )
    )
    model.fit(X_train, y_train)

    weights = np.asarray(
        [
            np.dot(dual_coef, support_vector).reshape(-1)
            for dual_coef, support_vector in zip(model.dual_coefs_, model.support_vectors_)
        ],
        dtype=float,
    )

    beta = []
    for feature_idx in range(weights.shape[1]):
        beta_i = 0.0
        for row_idx, class_idx in enumerate(y_cal_idx):
            lambda_term = lambda_ * weights[class_idx, feature_idx] * X_cal[row_idx, feature_idx]
            competing_weight = np.sum(np.delete(weights, class_idx, axis=0)[:, feature_idx])
            sum_term = lambda_p * competing_weight * X_cal[row_idx, feature_idx]
            beta_i -= lambda_term - sum_term
        beta.append(float(beta_i))

    return np.asarray(beta, dtype=float)
