# Minmial Strangeness Maximal Relevance Minimal Redundancy method.

We provide the novel MinStra_MaxRel_MinRed (MS_mRMR) feature selection method.
*MS_mRMR*  is the first feature selection method that mixed the concept of strangeness minimization [2] of the Conformal Prediction framework [1] with the feature selection based on information theory (Mutual information). *MS_mRMR*´s objective is minimizing the non-conformity of the features while maximizes the Conditional mutual information. Multiclass classficcation is available.  

We also provide the well known mRMR, JMI and relaxMRMR methods implemented in a fast way. The MI estimation is an adapted version of the provided by ...


## Requirements

Python 3.7 +
Scikit-learnt 1.2.2+
Numpy 
Scipy 



## Quickstart

A basic example is coded in *Library/example.ipynb*.

## References 

[1] V. Balasubramanian, S.-S. Ho, and V. Vovk, Conformal Prediction
for Reliable Machine Learning: Theory, Adaptations and Applications,
1st ed. San Francisco, CA, USA: Morgan Kaufmann Publishers Inc.,
2014.

[2] T. Bellotti, Z. Luo, and A. Gammerman, “Strangeness Minimisation
Feature Selection with Confidence Machines,” in Intelligent Data Engineering
and Automated Learning IDEAL 2006, ser. Lecture Notes in
Computer Science, E. Corchado, H. Yin, V. Botti, and C. Fyfe, Eds.
Berlin, Heidelberg: Springer, 2006, pp. 978–985.

