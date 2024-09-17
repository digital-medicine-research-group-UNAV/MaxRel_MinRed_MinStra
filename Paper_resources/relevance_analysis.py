

import os
import pandas as pd
import numpy as np
import itertools
import scipy
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
#from matplotlib_venn import venn3,venn2

import pickle
import argparse




def plot_comparison(OUT, list_of_names, N_classes):

    # read the dictionary from the file
    with open( OUT +'.pickle', 'rb') as f:
        OUT = pickle.load(f)



    data_mean = {}
    data_std = {}

    for i in range(0, len( list_of_names)):
        data_mean[list_of_names[i]] = []
        data_std[list_of_names[i]] = []

    for ele in OUT:

        keys = list(ele.keys())
        values = list(ele.keys())

        x = [*range(1 , len(ele["inefficiency"]) + 1)]

        #for i in range(len(values[0])):

        for name in list_of_names:
            
            data_mean[name] += [ele[name]]
            #if name != "inefficiency":
                
            #    data_mean[name] += [ele[name]]
            #else:
                
            #    data_mean[name] += [MinMaxscaler(ele[name], 1, N_classes)]
                
    return data_mean, data_std, x



parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True, help='dataset name.')
parser.add_argument('--methods', nargs='+', type=str, required=True, help='"mRMR, mRMR_MS, JMI"')
parser.add_argument('--classifiers', nargs='+', type=str, required=True, help='"SVM, KNN"')

# Parsear los argumentos
args = parser.parse_args()

dataset_user = args.dataset 
N_of_classes = args.n_classes
n_features = args.n_features
methods = args.methods
classifiers = args.classifiers

save_path = 'save/save_'+ dataset_user + '/'+ "Best_conformal_scores"+ dataset_user + ".csv"



dataset = dataset_user

list_of_names = [ "inefficiency", "P", "certainty", "P", "uncertainty", "P", "mistrust", "P",]

df = pd.DataFrame(columns=list_of_names)

list_of_names = [ "coverage","inefficiency", "certainty",  "uncertainty",  "mistrust", "S_score", "F_score", "Creditibily"]

for classifier in classifiers:

    for method in args.methods:

        method_file = "save/save_"+ dataset +"/"+ dataset +"_" +  method + "_conformal_" + classifier 

        #"..\mRMR-MS_new\mRMR-MS_new\save\save_breast\\breast_mRMR-MS_tangent_conformal_XGBOOST"

    


        if dataset == "subiculum":
            subi = 1
        else:
            subi = 0


        data_mean, data_std, x_1 = plot_comparison(method_file, list_of_names, N_of_classes )



        list_of_values = []
        for name in list_of_names:

            if name in ["coverage", "S_score", "F_score", "Creditibily"]:
                continue

        
            mean = np.mean(np.array(data_mean[name]), axis=0).tolist()
            std_dev = np.std(np.array(data_mean[name]), axis=0).tolist()

        
    
            alfa = 0.05

            if name == "inefficiency": 

                if subi == 1:
                # Specify the target number
                    target_number = 1.0

                    mean=np.array(mean)
            
            
                    nearest_value = mean[np.where(mean<=1.0)[0][0]-1]
                    nearest_position = np.where(mean<=1.0)[0][0]-1


                    #print(name, ": ", round(nearest_value,2), nearest_position + 1)
                    list_of_values.append(round(nearest_value,2))
                    list_of_values.append(int(nearest_position + 1))

                else:
                    #print(name, ": ", round(np.min(mean),2), np.argmin(mean) + 1)
                    list_of_values.append(round(np.min(mean),2))
                    list_of_values.append(int(np.argmin(mean) + 1))

                    #t_stat, p_valor = stats.ttest_ind(mean_1, mean_2, alternative = "less")
    
            elif name == "certainty":
                #print(name, ": ", round(np.max(mean),2), np.argmax(mean)+1)
                list_of_values.append(round(np.max(mean),2))
                list_of_values.append(int(np.argmax(mean)+1))
                #t_stat, p_valor = stats.ttest_ind(mean_1, mean_2,  alternative = "less")

            elif name == "uncertainty":
                #t_stat, p_valor = stats.ttest_ind(mean_1, mean_2, alternative = "less")
                #print(name, ": ", round(np.min(mean),2), np.argmin(mean)+1)
                list_of_values.append(round(np.min(mean),2))
                list_of_values.append(int(np.argmin(mean)+1))

            elif name == "mistrust":
                #print(name, ": ", round(np.max(mean),2), np.argmax(mean)+1)
                list_of_values.append(round(np.max(mean),2))
                list_of_values.append(int(np.argmax(mean)+1))
                #t_stat, p_valor = stats.ttest_ind(mean_1, mean_2, alternative = "less")

            else:
                pass
        
        new_series = pd.Series(list_of_values, index=df.columns, name=method + "-" + classifier)

        # Concatenar la nueva serie al DataFrame
        df = pd.concat([df, new_series.to_frame().T], ignore_index=False)
    

print(df)


df.to_csv(save_path, index=True, header=True)
    

