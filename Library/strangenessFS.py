

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC



# Let adapt the OneVsRestClassifier to extract the support vectors and coefficients of the SVM model

class CustomOneVsRestClassifier(OneVsRestClassifier):
    def __init__(self, estimator):
        super().__init__(estimator)
        
    def fit(self, X, y):

        self.classes_ = np.unique(y)    
        
        self.dual_coefs_ = []
        self.support_vectors_ = []
        self.estimators_ = []
        for i, class_ in enumerate(self.classes_):
            
            y_binary = np.where(y == class_, 1, -1)
            estimator = self.estimator.fit(X, y_binary)
            
            self.dual_coefs_.append(estimator.dual_coef_)
            self.support_vectors_.append(estimator.support_vectors_) 
            self.estimators_.append(estimator)
        
        return self


# Main funtion of this script

def beta_measures(X, Y, classes_, split_size = 0.5, lambda_= 0.5, kernel = 'linear') -> list:

    X_tr, X_cal,  Y_tr, Y_cal = train_test_split(X, Y, test_size = split_size, stratify=Y)
    

    list_of_index: list = np.arange(len(X_tr[0])).tolist()   # Empieza en 0   # Array must be a list
    n: int = len(list_of_index)                              # counter of features
    Lambda_p = (1-lambda_) / (len(classes_)-1)


    if len(classes_) == 2:
            
            Y_tr = np.where(Y_tr == 0, -1, 1)
            Y_cal = np.where(Y_cal == 0, -1, 1)
            
            # Create and train the SVM model with RBF kernel
            model = SVC(kernel=kernel, degree=3, tol =1e-3, probability=True,  cache_size=900, C=1.0, max_iter=900000)
            model.fit(X_tr, Y_tr)

            # Get support vectors and coefficients (weights in the transformed space)
            support_vectors = model.support_vectors_
            dual_coef = model.dual_coef_

            w = np.dot(dual_coef, support_vectors)  # approximate weights
                
            beta = []
            for j in range(n):

                beta_i = 0
                for i in range(len(X_cal)):
                        
                    #print(Y_cal[i])
                    beta_ij = w[0][j] * Y_cal[i] * X_cal[i][j]
                    beta_i = beta_i - beta_ij

                    i = i + 1

                beta.append(beta_i)

    else:
        
        # Create and train the custom One-vs-Rest SVM model with RBF kernel
        model = CustomOneVsRestClassifier(SVC(kernel=kernel, degree=3,  tol =1e-3, probability=True, cache_size=900, C=1.0, max_iter=900000 ))
        model.fit(X_tr, Y_tr)

        # Extract support vectors and coefficients for each class
        support_vectors = model.support_vectors_ 
        dual_coefs = model.dual_coefs_
        

        # Approximate feature weights for each class
        
        

        w = [np.dot(dual_coef, support_vector).flatten().tolist() for dual_coef, support_vector in zip(dual_coefs, support_vectors)]
        
        w: np.ndarray = np.array(w)
        X_cal = np.array(X_cal)
        Y_cal = np.array(Y_cal)
        beta:list = []
        
        for j in range(w.shape[1]):
                
            #print(f"Iteración {j} de {w.shape[1]}")
            beta_i = 0
            for i in range(len(X_cal)):
                    
                lambda_term = lambda_ * w[Y_cal[i], j] * X_cal[i, j]
 
                sum_term = Lambda_p * np.sum(np.delete(w, Y_cal[i], axis=0)[:, j]) * X_cal[i, j].sum()
                    
                beta_ij = lambda_term - sum_term
                beta_i = beta_i - beta_ij


            beta.append(beta_i)
            
    
    
    return beta