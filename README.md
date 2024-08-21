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


