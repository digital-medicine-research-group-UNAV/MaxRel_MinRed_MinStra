# Minimal Strangeness Maximal Relevance Minimal Redundancy method.

We provide the novel MinStra_MaxRel_MinRed (MS_mRMR) feature selection method.
*MS_mRMR*  is the first feature selection method that mixed the concept of strangeness minimization [1] of the Conformal Prediction framework [2] with the feature selection based on information theory (Mutual information). *MS_mRMR*´s objective is minimizing the non-conformity of the features while maximizes the Conditional mutual information. This helps to compute more efficient prediction sets.  

Our library also provides the well known mRMR, JMI and relaxMRMR methods implemented in a fast way. The mutual information estimation is an adapted version of the methodology provided in [3].


## Requirements

Python 3.7 +
Scikit-learnt 1.2.2+
Numpy <2.0.0
Scipy 



## Quickstart

A basic example is coded at *Library/example.ipynb*.

```python
import numpy as np
from sklearn.datasets import make_classification
from main import FeatureSelector

# Create a synthetic classification dataset

X_tr: np.ndarray
Y_tr: np.ndarray
n_informative: int = 15
n_classes: int = 3

X_tr, Y_tr = make_classification(
    n_samples=300,    
    n_features=30,     
    n_informative=n_informative,  
    n_redundant=5,     
    n_classes=n_classes)  

```

```python
sele = FeatureSelector(classes_ = [i for i in range(n_classes)],
                       max_features = n_informative,
                       parallel= True,
                       verbose = True)  


kernel: str = "linear"   # linear is only for mRMR_MS. Other options are "rbf" and "poly".
split_size: float = 0.5  # is only for mRMR_MS.
sele.mRMR_MS(X_tr, Y_tr,  kernel, split_size) # compute the mRMR_MS feature selection.
print(sele.all_selected_features[-1])

sele.mRMR(X_tr, Y_tr)                       # compute the mRMR feature selection.
print(sele.all_selected_features[-1])

sele.JMI(X_tr, Y_tr)                        # compute the JMI feature selection.
print(sele.all_selected_features[-1])

sele.relax_mRMR(X_tr, Y_tr)                 # compute the relax_mRMR feature selection.
print(sele.all_selected_features[-1])

```

## References 

[1] T. Bellotti, Z. Luo, and A. Gammerman, “Strangeness Minimisation 
Feature Selection with Confidence Machines,” in Intelligent Data Engineering
and Automated Learning IDEAL 2006, ser. Lecture Notes in
Computer Science, E. Corchado, H. Yin, V. Botti, and C. Fyfe, Eds.
Berlin, Heidelberg: Springer, 2006, pp. 978–985. 2014.

[2] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.

[3] O. C. Mesner and C. R. Shalizi, "Conditional Mutual Information Estimation for Mixed, Discrete and Continuous Data," in IEEE Transactions on Information Theory, vol. 67, no. 1, pp. 464-484, Jan. 2021, doi: 10.1109/TIT.2020.3024886.


