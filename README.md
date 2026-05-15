# MaxRel-MinRed-MinStra

This repository provides an implementation of **MaxRel-MinRed-MinStra**
(`mRMR_MS`), a feature-selection criterion that combines information-theoretic
maximal relevance/minimal redundancy with a minimal-strangeness term inspired by
conformal prediction.

The library currently exposes four methods:

- `mRMR_MS`: Maximal Relevance, Minimal Redundancy, Minimal Strangeness.
- `mRMR`: maximal relevance and minimal redundancy.
- `JMI`: joint mutual information.
- `relax_mRMR`: relaxed mRMR with conditional redundancy terms.

The code in `Library/` is small and package-oriented. 


Main dependencies are `numpy`, `pandas`, `scipy` and `scikit-learn`.

## Quickstart

```python
import numpy as np
from sklearn.datasets import make_classification

from maxrel_minred_minstra import FeatureSelector

X, y = make_classification(
    n_samples=300,
    n_features=30,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=123,
)

selector = FeatureSelector(
    classes_=np.unique(y),
    max_features=15,
    kernel="linear",
    random_state=123,
)

selected_path = selector.mRMR_MS(X, y)
print(selected_path[-1])
```

Each method returns a cumulative selection path:

```python
[
    [f1],
    [f1, f2],
    [f1, f2, f3],
    ...
]
```

The final subset is also available as:

```python
selector.selected_features_
```

The notebook `Library/example.ipynb` contains a compact example for all four
methods.

## Class

```python
selector = FeatureSelector(
    classes_=None,
    lambda_=0.5,
    max_features=-1,
    kernel="linear",
    split_size=0.5,
    k=3,
    random_state=None,
    verbose=False,
)
```

Available calls:

```python
selector.mRMR_MS(X, y)
selector.mRMR(X, y)
selector.JMI(X, y)
selector.relax_mRMR(X, y)
```

`X` must be a two-dimensional array-like object and `y` a one-dimensional class
label vector. Feature indices in the returned path refer to the columns of `X`.



## References

[1] T. Bellotti, Z. Luo, and A. Gammerman, "Strangeness Minimisation Feature
Selection with Confidence Machines," in *Intelligent Data Engineering and
Automated Learning IDEAL 2006*, Lecture Notes in Computer Science, pp. 978-985.

[2] V. Balasubramanian, S.-S. Ho, and V. Vovk, *Conformal Prediction for
Reliable Machine Learning: Theory, Adaptations and Applications*, Morgan
Kaufmann.

[3] O. C. Mesner and C. R. Shalizi, "Conditional Mutual Information Estimation
for Mixed, Discrete and Continuous Data," *IEEE Transactions on Information
Theory*, vol. 67, no. 1, pp. 464-484, Jan. 2021.
