
import numpy as np 

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse
import json 

from conf_stability import get_stability
from scipy import stats





def list_maker(*data_, data_21, data_42, data_64, data_128, eje ):
    
    data = []
    for ele in data_:
        data = data +  [ data_[eje][i] for i in range(len(ele[0])) ]

    Data = np.array(data)
    Data = np.sum(Data, axis=0)/len(data_)
    
    return Data.tolist()

def MinMaxscaler(x, x_min, x_max):
    return [(x[i] - x_min) / (x_max - x_min) for i in range(len(x))]

list_of_names = [ "coverage", "inefficiency", 
                    "certainty", "uncertainty", "mistrust",
                    "S_score", "F_score", "Creditibily"]


def plot_comparison(OUT, list_of_names, N_classes):

    # read the dictionary from the file
    with open( OUT, 'rb') as f:
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
            

            if name != "inefficiency":
                
                data_mean[name] += [ele[name]]
            else:
                
                data_mean[name] += [ele[name]] #[MinMaxscaler(ele[name], 1, N_classes)]
                
    return data_mean, data_std, x

class plotter:
    
    #def __init__(self):
        # Constructor code here

    def plot_conformal(self, dataset, method1, method2, classifier, N_of_classes ):

        #dataset = "IMV_ABA"
        #classifier = "SVM"
        #N_of_classes = 3

        save_path = 'save/save_'+ dataset + '/plots/'+ "conformal_scores_"+ dataset + "_" + classifier + "_" + method1 + "_" +  method2 + '.pdf' 
        RFE_file =  "save/save_"+ dataset +"/"+ dataset +"_" +  method1 + "_conformal_" + classifier + ".pickle"
        CRFE_file = "save/save_"+ dataset +"/"+ dataset +"_" +  method2 + "_conformal_" + classifier + ".pickle"

        name_a = ' (a) ' + method1
        name_b = ' (b) ' + method2

        list_of_colors = ["blue", "red", "green", "orange", "black" ]
        list_of_styles = ['-','--', '-.', ":" , ":"]
        list_of_dots = ["*", "+", "*", "+", "*"]
        list_of_w = [1.2, 1.9, 1.5, 1.9, 1.5]
        list_of_dashes = [(1,0), (6,1), (1,2), (4,1,1,1), (3,1,1,1,1,1)]
        list_of_intensities = [0.40 , 0.50 , 0.20 , 0.35 , 0.19]

        data_mean, data_std, x = plot_comparison(RFE_file, list_of_names, N_of_classes )
        
        with open("save/save_"+ dataset +"/"+ dataset +"_" +  method1 + "_conformal_" + classifier + ".txt", 'w') as file:
            json.dump(data_mean, file)

        with open("save/save_"+ dataset +"/"+ dataset +"_" +  method1 + "_conformal_" + classifier + "_std.txt", 'w') as file:
            json.dump(data_std, file)

        #fig = plt.figure(figsize=(4, 3), dpi=300)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
     
        i = 0
        #x = x[:zoom]
        for name in list_of_names:

            mean = np.mean(np.array(data_mean[name]), axis=0).tolist()
            std_dev = np.std(np.array(data_mean[name]), axis=0).tolist()

            data_mean[name] = mean  #[num for idx, num in enumerate(mean) if idx % 2 == 0] #mean
            data_std[name] = std_dev #[num for idx, num in enumerate(std_dev) if idx % 2 == 0]  #std_dev

            

            if name in ["S_score", "F_score", "Creditibily"]:
                pass

            else:
        
                rev_list = data_mean[name]
                #rev_list.reverse()
                rev_std = data_std[name]
                #rev_std.reverse()
                ax1.plot(x, rev_list,
                    label = name, dashes=list_of_dashes[i],
                    markersize=2.5, color = list_of_colors[i],  
                    linestyle="--", linewidth=list_of_w[i])
        
       
        
                ax1.fill_between(x, np.array(rev_list) + np.array(rev_std),
                                np.array(rev_list) - np.array(rev_std), 
                                color= list_of_colors[i], alpha=list_of_intensities[i])
        
            i = i + 1

            #ax1.axvline(x=zoom, color='black', linestyle='--')

        ax1.set_xlabel("Num. of Features")
        ax1.set_ylabel("Score")


        y_min, y_max = ax1.get_ylim()
        y_padding = 0.00005 * (y_max - y_min)
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)

        # Add grid lines
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.set_yticks([i/10 for i in range(11)])  # Set the y ticks at 0, 0.1, 0.2, ..., 1

        
        
 
        data_mean, data_std, x = plot_comparison(CRFE_file, list_of_names,  N_of_classes)

        with open("save/save_"+ dataset +"/"+ dataset +"_" +  method2 + "_conformal_" + classifier + ".txt", 'w') as file:
            json.dump(data_mean, file)
        
        with open("save/save_"+ dataset +"/"+ dataset +"_" +  method2 + "_conformal_" + classifier + "_std.txt", 'w') as file:
            json.dump(data_std, file)

        #data_mean, data_std, x = data_mean[0:100], data_std[0:100], x[0:100]
        #x = x[0:zoom]
        #x = x[:zoom]
        i = 0
        for name in list_of_names:

            mean = np.mean(np.array(data_mean[name]), axis=0).tolist()#[:zoom]
            std_dev = np.std(np.array(data_mean[name]), axis=0).tolist()#[:zoom]
    
            data_mean[name] = mean #  [num for idx, num in enumerate(mean) if idx % 2 != 0] #mean
            #std_dev = sem(result)
            data_std[name] = std_dev #[num for idx, num in enumerate(std_dev) if idx % 2 != 0] #std_dev
    
            #print(name, data_mean[name][len(data_mean[name])-15])  ???

            if name in ["S_score", "F_score", "Creditibily"]:
                pass

            else:
        
                rev_list = data_mean[name]
                #rev_list.reverse()
                rev_std = data_std[name]
                #rev_std.reverse()
                ax2.plot(x, rev_list,
                        label = name, dashes=list_of_dashes[i],
                        markersize=2.5, color = list_of_colors[i],  
                        linestyle="--", linewidth=list_of_w[i])
        
                
                ax2.fill_between(x, np.array(rev_list) + np.array(rev_std),
                                np.array(rev_list) - np.array(rev_std), 
                                color= list_of_colors[i], alpha=list_of_intensities[i])
        
            i = i + 1

        

        ax2.set_xlabel("Num. of Features")
        ax2.set_ylabel("Score")

        y_min, y_max = ax2.get_ylim()
        y_padding = 0.00005 * (y_max - y_min)
        ax2.set_ylim(y_min - y_padding, y_max + y_padding)

        # Add grid lines
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.set_yticks([i/10 for i in range(11)])  # Set the y ticks at 0, 0.1, 0.2, ..., 1

        ax1.set_title(name_a)
        ax2.set_title(name_b)

        handles, labels = [], []

        h, l = ax2.get_legend_handles_labels()
        handles += h
        labels += l

        fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, handlelength=3.8, fontsize=11, bbox_to_anchor=(0.5, -0.14))

        plt.savefig( save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

        return None

    def plot_stability_comp(self, dataset, method1, method2, method3, method4, method5, method6, n_features, alpha = 0.1):

        #dataset = "subiculum"

        save_path = 'save/save_'+ dataset + '/plots/'+ "stability_"+ dataset +  '.pdf'
        

        path = "save/save_"+ dataset +"/"+ dataset +"_" + method1 + "_conformal_SVM.pickle"

        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df1 = pd.DataFrame(columns=Columns)
 

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
            
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 500)
    
            if set_pre == []:
                set_pre = [-.5, 1.]
    
            df1.loc[len(df1)] = [set_pre[0], set_pre[1], nogue]


        path = "save/save_"+ dataset +"/"+ dataset +"_" + method2 + "_conformal_SVM.pickle"
 
        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df2 = pd.DataFrame(columns=Columns)

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
        
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 1000)

            if set_pre == []:
                set_pre = [-.5, 1.]

            df2.loc[len(df2)] = [set_pre[0], set_pre[1], nogue]

        
        path = "save/save_"+ dataset +"/"+ dataset +"_" + method3 + "_conformal_SVM.pickle"
 
        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df3 = pd.DataFrame(columns=Columns)

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
        
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 1000)

            if set_pre == []:
                set_pre = [-.5, 1.]

            df3.loc[len(df3)] = [set_pre[0], set_pre[1], nogue]


        path = "save/save_"+ dataset +"/"+ dataset +"_" + method4 + "_conformal_SVM.pickle"
 
        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df4 = pd.DataFrame(columns=Columns)

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
        
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 1000)

            if set_pre == []:
                set_pre = [-.5, 1.]

            df4.loc[len(df4)] = [set_pre[0], set_pre[1], nogue]



        path = "save/save_"+ dataset +"/"+ dataset +"_" + method5 + "_conformal_SVM.pickle"
 
        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df5 = pd.DataFrame(columns=Columns)

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
        
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 1000)

            if set_pre == []:
                set_pre = [-.5, 1.]

            df5.loc[len(df5)] = [set_pre[0], set_pre[1], nogue]


        path = "save/save_"+ dataset +"/"+ dataset +"_" + method6 + "_conformal_SVM.pickle"
 
        Columns = ['Low_int' ,'High_int', 'Nogue']

        # Create the dataframe
        df6 = pd.DataFrame(columns=Columns)

        with open(path, 'rb') as f:
            OUT = pickle.load(f)

        feat_subsets = [ele["Index"] for ele in OUT ] 

        for i in range(len(feat_subsets[0])):

            data_train = []

            for subset in feat_subsets:
        
                ele_matrix = [0 if j not in subset[i] else 1 for j in range(n_features)] ## modificar esto, esta mal hacer cohompresinson y marcar 1 donde exista feature 
        
                data_train.append(ele_matrix) 
    
            set_pre, nogue = get_stability( alpha, np.array(data_train), M = -1, lin_spaces = 1000)

            if set_pre == []:
                set_pre = [-.5, 1.]

            df6.loc[len(df6)] = [set_pre[0], set_pre[1], nogue]



        plt.figure(figsize=(10, 5))



        plt.plot(df1.index.tolist(), df1["Nogue"], label =  method1, markersize=2.5, color = "blue",  
                linestyle="-")
    
        plt.fill_between(df1.index.tolist(),   df1['High_int'],
                            df1['Low_int'], 
                            color= "blue", alpha=.33)


        plt.plot(df2.index.tolist(), df2["Nogue"], label = method2, markersize=2.5, color = "red",  
                linestyle=":")
    
        plt.fill_between(df2.index.tolist(),   df2['High_int'],
                            df2['Low_int'], 
                            color= "red", alpha=.30)
        

        plt.plot(df3.index.tolist(), df3["Nogue"], label = method3, markersize=2.5, color = "purple",  
                linestyle="-.")
    
        plt.fill_between(df3.index.tolist(),   df3['High_int'],
                            df3['Low_int'], 
                            color= "purple", alpha=.25)
        

        plt.plot(df4.index.tolist(), df4["Nogue"], label = method4, markersize=2.5, color = "black",  
                linestyle="--")

        plt.fill_between(df4.index.tolist(), df4['High_int'],
                            df4['Low_int'], 
                            color= "black", alpha=.27)
        
        plt.plot(df5.index.tolist(), df5["Nogue"], label = method5, markersize=2.5, color = "green",  
                linestyle="--")

        plt.fill_between(df5.index.tolist(), df5['High_int'],
                            df5['Low_int'], 
                            color= "green", alpha=.36)
        
        plt.plot(df6.index.tolist(), df6["Nogue"], label = method6, markersize=2.5, color = "orange",  
                linestyle="-")

        plt.fill_between(df6.index.tolist(), df6['High_int'],
                            df6['Low_int'], 
                            color= "orange", alpha=.26)


        plt.xlabel("Num. of Features")
        plt.ylabel("Stability")


        y_min, y_max = -0.5, 1
        y_padding = 0.05 * (y_max - y_min)
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.yticks([i/10 for i in range(-5,11)])  # Set the y ticks at 0, 0.1, 0.2, ..., 1
        
        plt.legend(loc='lower center', ncol=4, fontsize=13)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Show the plot
        #plt.show()

        df1.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method1 +  '.csv', index=False)
        df2.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method2 +  '.csv', index=False)
        df3.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method3 +  '.csv', index=False)
        df4.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method4 +  '.csv', index=False)
        df5.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method5 +  '.csv', index=False)
        df6.to_csv('save/save_'+ dataset +  '/STABILITY_'+ dataset + '_' + method6 +  '.csv', index=False)

        # Python

        t_statistic41, p_value41 = stats.ttest_ind(df4["Nogue"], df1["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic42, p_value42 = stats.ttest_ind(df4["Nogue"], df2["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic43, p_value43 = stats.ttest_ind(df4["Nogue"], df3["Nogue"],nan_policy="omit", alternative = "greater")

        t_statistic51, p_value51 = stats.ttest_ind(df5["Nogue"], df1["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic52, p_value52 = stats.ttest_ind(df5["Nogue"], df2["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic53, p_value53 = stats.ttest_ind(df5["Nogue"], df3["Nogue"],nan_policy="omit", alternative = "greater")

        t_statistic61, p_value61 = stats.ttest_ind(df6["Nogue"], df1["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic62, p_value62 = stats.ttest_ind(df6["Nogue"], df2["Nogue"],nan_policy="omit", alternative = "greater")
        t_statistic63, p_value63 = stats.ttest_ind(df6["Nogue"], df3["Nogue"],nan_policy="omit", alternative = "greater")

        with open('save/save_'+ dataset +  '/STABILITY_'+ dataset +  '.txt', 'w') as doc:
            doc.write(f"Method: {method4}, {method1}. t-statistic: {t_statistic41}, p-value: {p_value41}")
            doc.write(f"\nMethod: {method4}, {method2}. t-statistic: {t_statistic42}, p-value: {p_value42}")
            doc.write(f"\nMethod: {method4}, {method3}. t-statistic: {t_statistic43}, p-value: {p_value43}")

            doc.write(f"\nMethod: {method5}, {method1}. t-statistic: {t_statistic51}, p-value: {p_value51}")
            doc.write(f"\nMethod: {method5}, {method2}. t-statistic: {t_statistic52}, p-value: {p_value52}")
            doc.write(f"\nMethod: {method5}, {method3}. t-statistic: {t_statistic53}, p-value: {p_value53}")

            doc.write(f"\nMethod: {method6}, {method1}. t-statistic: {t_statistic61}, p-value: {p_value61}")
            doc.write(f"\nMethod: {method6}, {method2}. t-statistic: {t_statistic62}, p-value: {p_value62}")
            doc.write(f"\nMethod: {method6}, {method3}. t-statistic: {t_statistic63}, p-value: {p_value63}")


        return None
    




parser = argparse.ArgumentParser()
parser.add_argument('--n_features', type=int, required=True)
parser.add_argument('--n_classes', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True, help='dataset name.')
parser.add_argument('--methods', nargs='+', type=str, required=False, help='"mRMR, mRMR_MS,relax_mRMR, JMI"')

# Parsear los argumentos
args = parser.parse_args()

dataset_user = args.dataset 
n_classes = args.n_classes
n_features = args.n_features
methods = args.methods


if methods == None:

    plotter().plot_stability_comp( dataset_user, "mRMR", "JMI", "relax_mRMR","mRMR_MS_linear", "mRMR_MS_rbf", "mRMR_MS_poly", n_features, alpha = 0.1)

    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_linear","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_linear","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_linear","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_linear","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_linear","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_linear","KNN", n_classes )


    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_rbf","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_rbf","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_rbf","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_rbf","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_rbf","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_rbf","KNN", n_classes )


    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_poly","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "mRMR", "mRMR_MS_poly","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_poly","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "JMI", "mRMR_MS_poly","KNN", n_classes )

    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_poly","SVM", n_classes )
    plotter().plot_conformal(dataset_user, "relax_mRMR", "mRMR_MS_poly","KNN", n_classes )
 
    
else:
    for method in args.methods:

        plotter().plot_conformal(dataset_user, method, "mRMR_MS","SVM", n_classes )
        plotter().plot_conformal(dataset_user, method, "mRMR_MS","KNN", n_classes)

        plotter().plot_stability_comp( dataset_user, n_features, alpha = 0.1)




