from __future__ import annotations

import numpy as np
import pandas as pd

from ._cmi import cmi
from ._strangeness import strangeness_scores
from ._utils import min_max_normalize, validate_X_y
epsilon = 1e-9

class FeatureSelector:
    """Feature selector for mRMR_MS, mRMR, JMI and relax_mRMR.

    The methods return a cumulative list of selected feature indices:
    ``[[f1], [f1, f2], ..., [f1, ..., fk]]``.
    """

    def __init__(
        self,
        classes_=None,
        lambda_: float = 0.5,
        max_features: int = -1,
        kernel: str = "linear",
        split_size: float = 0.5,
        k: int = 3,
        random_state: int | None = None,
        verbose: bool = False,
        parallel: bool | None = None,
    ):
        self.classes_ = None if classes_ is None else np.asarray(classes_)
        self.lambda_ = float(lambda_)
        self.max_features = int(max_features)
        self.kernel = "linear" if kernel in [None, ""] else str(kernel)
        self.split_size = float(split_size)
        self.k = int(k)
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.parallel = parallel
        self.all_selected_features: list[list[int]] = []
        self.selected_features_: list[int] = []

    def _prepare(self, X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, int]:
        X, y = validate_X_y(X, y)
        classes = np.unique(y) if self.classes_ is None else np.asarray(self.classes_)
        if set(np.unique(y)) - set(classes):
            raise ValueError("classes_ must contain all class labels present in y.")

        n_features = X.shape[1]
        max_features = n_features if self.max_features == -1 else min(self.max_features, n_features)
        if max_features < 1:
            raise ValueError("max_features must be -1 or a positive integer.")

        data = pd.DataFrame(X, dtype=np.float32)
        data["Class"] = y
        return X, y, classes, data, max_features

    def _log(self, step: int, total: int) -> None:
        if self.verbose:
            print(f"{step + 1} out of {total} features were selected", flush=True)

    def _relevance(self, feature: int, data: pd.DataFrame) -> float:
        return cmi([feature], ["Class"], [], self.k, data)

    def _redundancy(self, feature: int, selected: list[int], data: pd.DataFrame) -> float:
        if not selected:
            return 0.0
        return float(np.mean([cmi([selected_feature], [feature], [], self.k, data) for selected_feature in selected]))

    def _conditional_redundancy(self, feature: int, selected: list[int], data: pd.DataFrame) -> float:
        if not selected:
            return 0.0
        return float(
            np.mean([cmi([selected_feature], [feature], ["Class"], self.k, data) for selected_feature in selected])
        )

    def _third_order_conditional_redundancy(self, feature: int, selected: list[int], data: pd.DataFrame) -> float:
        if len(selected) <= 1:
            return 0.0

        values = []
        for conditioning_feature in selected:
            inner = [
                cmi([feature], [other_feature], [conditioning_feature], self.k, data)
                for other_feature in selected
                if other_feature != conditioning_feature
            ]
            if inner:
                values.append(float(np.mean(inner)))

        return float(np.mean(values)) if values else 0.0

    def _score_feature(self, feature: int, selected: list[int], data: pd.DataFrame, method: str) -> float:
        relevance = self._relevance(feature, data)

        if method == "mRMR":
            return relevance - self._redundancy(feature, selected, data)
        if method == "JMI":
            return (
                relevance
                - self._redundancy(feature, selected, data)
                + self._conditional_redundancy(feature, selected, data)
            )
        if method == "relax_mRMR":
            return (
                relevance
                - self._redundancy(feature, selected, data)
                + self._conditional_redundancy(feature, selected, data)
                - self._third_order_conditional_redundancy(feature, selected, data)
            )

        raise ValueError(f"Unknown method: {method}")

    def _select_information_method(self, X, y, method: str) -> list[list[int]]:
        _, _, _, data, max_features = self._prepare(X, y)
        selected: list[int] = []
        remaining = list(range(data.shape[1] - 1))
        all_selected: list[list[int]] = []

        for step in range(max_features):
            self._log(step, max_features)
            if step == 0:
                scores = [self._relevance(feature, data) for feature in remaining]
            elif method == "relax_mRMR" and len(selected) == 1:
                scores = [self._score_feature(feature, selected, data, "JMI") for feature in remaining]
            else:
                scores = [self._score_feature(feature, selected, data, method) for feature in remaining]

            accepted_position = int(np.argmax(scores))
            selected_feature = remaining.pop(accepted_position)
            selected.append(selected_feature)
            all_selected.append(selected.copy())

        self.all_selected_features = all_selected
        self.selected_features_ = selected.copy()
        return all_selected

    def mRMR_MS(self, X, y, kernel: str | None = None, split_size: float | None = None) -> list[list[int]]:
        X, y, classes, data, max_features = self._prepare(X, y)
        kernel = self.kernel if kernel in [None, ""] else str(kernel)
        split_size = self.split_size if split_size is None else float(split_size)

        selected: list[int] = []
        remaining = list(range(X.shape[1]))
        all_selected: list[list[int]] = []

        for step in range(max_features):
            self._log(step, max_features)
            if step == 0:
                phi = np.asarray([self._relevance(feature, data) for feature in remaining], dtype=float)
            else:
                phi = np.asarray(
                    [self._score_feature(feature, selected, data, "mRMR") for feature in remaining],
                    dtype=float,
                )

            beta = strangeness_scores(
                X[:, remaining],
                y,
                classes=classes,
                split_size=split_size,
                lambda_=self.lambda_,
                kernel=kernel,
                random_state=self.random_state,
            )

            phi = min_max_normalize(phi,  lower=epsilon, upper=1.0)
            beta = min_max_normalize(beta,  lower=epsilon, upper=1.0)
            alpha = phi / beta

            accepted_position = int(np.argmax(alpha))
            selected_feature = remaining.pop(accepted_position)
            selected.append(selected_feature)
            all_selected.append(selected.copy())

        self.all_selected_features = all_selected
        self.selected_features_ = selected.copy()
        return all_selected

    def mRMR(self, X, y) -> list[list[int]]:
        return self._select_information_method(X, y, "mRMR")

    def JMI(self, X, y) -> list[list[int]]:
        return self._select_information_method(X, y, "JMI")

    def relax_mRMR(self, X, y) -> list[list[int]]:
        return self._select_information_method(X, y, "relax_mRMR")
