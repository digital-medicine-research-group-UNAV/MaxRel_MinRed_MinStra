




import numpy as np 
import matplotlib.pyplot as plt
import itertools
import random
from math import comb

from init import getStability

from sklearn.neighbors import KernelDensity








def folded_samples(data_cal, k):
    
    index_list = list(range(len(data_cal)))
    combinations = list(itertools.combinations(index_list, k))

    muestras_calibracion_data = []
    

    for combinacion in combinations:
        ele = [data_cal[combinacion[i]] for i in range(k)]
        #print(ele)
        muestras_calibracion_data.append(ele)

    #print(muestras_calibracion_data)  

    return muestras_calibracion_data



def generate_bernoulli_array(n, p):
    # Generate an array of n variables following a Bernoulli distribution with parameter p
    bernoulli_array = np.random.choice([0, 1], size=n, p=[1-p, p])
    return bernoulli_array



def generate_artificial_feature_selection(m,n, probabilities_train):

    p_0 = probabilities_train[0]
    arrays = [generate_bernoulli_array(m, p_0)]

    for i in range(1,n):

        p = probabilities_train[i]
        result_array = generate_bernoulli_array(m, p)
        arrays = np.append(arrays, [result_array], axis=0)


    result_array = np.column_stack(arrays)

    return result_array







## we pick the hyper-parameters to generate the aritifical dataset


n_examples = 1000
range_cover_conf = []
range_cover_nogue = []
range_len_conf = []
All_P_values = []
range_len_nogue = []
n_alphas = 10
Alpha = []
range_len_conf_std =[]
range_len_nogue_std =[]

# m_tr  # number of samples
 #d # number of features 
#M # folds for predict stability 
# lin_spaces


def get_stability(alpha,data_train, M = -1, lin_spaces = 500):

    Alpha = 1. - alpha

    m_tr = data_train.shape[0]
    d = data_train.shape[1]

        
    if M == -1:

        Combinations = []
        folds = []

        for m in range(2, m_tr):
            
            num_combinations = comb(len(data_train), m)
            Combinations.append(num_combinations)
            folds.append(m)

            max_value = max(Combinations)
           
            M_index = len(Combinations) - 1 - Combinations[::-1].index(max_value)
            
            M = folds[M_index] 
            

    else:
        pass

    #print('M:', M)
        
    data_train_sep = folded_samples(data_train, M)
        
    stability_measures = []
    for data in data_train_sep:
        stability_measures.append(getStability(data))

    #print(true)
    Y_trial = np.linspace(-1/(M-1), 1, num=lin_spaces)

        
    n_points = len(data_train_sep)
    limit = np.ceil((1.-alpha)*(n_points + 1))

    c = []
    P_values = []
    for y_trial in Y_trial:

        stability_measures_c = stability_measures.copy()

        # Multiply each element of stability_measures_c by a random number
        #stability_measures_c = [x * random.uniform(0, 0.0001) for x in stability_measures_c]
            
        stability_measures_c.append(y_trial)

        mean = np.mean(stability_measures_c)
        std = np.std(stability_measures_c)

        ncm = np.abs((stability_measures_c - mean)/std)
        ncm_n_1 = ncm[-1].copy()

        counter = 0
        tau = random.uniform(0,1)
        for measure in ncm:

            if measure < ncm_n_1:
                counter = counter + 1
            if measure == ncm_n_1:
                counter = counter + tau*1
            else:
                pass
            
        p_value = (counter)/(len(ncm) )
        P_values.append(p_value)
        
    intervalo = []
    for i in range(len(Y_trial)): 
    
        if (len(stability_measures) + 1)*P_values[i] <=np.ceil((1. - alpha)*(len(stability_measures) + 1)):            
            intervalo.append(Y_trial[i])
    
    try:
        set_prediction =  [intervalo[0],intervalo[-1]]  
    except IndexError:
        set_prediction = intervalo    

    return set_prediction, getStability(data_train)  # return the conformal interval and the nogueira stability measure of the original data
    
    









