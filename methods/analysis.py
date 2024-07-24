import matplotlib.pyplot as plt
from methods.sfranco_analysis import *
from methods.jrc_analysis import *
from utils.test_data import *
import numpy as np
import pandas as pd

def plot_classif_vs_noiselevel_unique_nonunique(plot_results,N_unique,N_nonunique,path=None):
    labelsize=14
    titlesize=16
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(6.5,2))
    markers = ['s','o','x','P','*']
    # Plot unique'
    fd_orders = plot_results['fd_order'].unique()
    plot_class = plot_results[plot_results['True_Class'] == "unique"]
    for i,fd_order in enumerate(fd_orders):
            N=N_unique
            subset = plot_class[plot_results['fd_order'] == fd_order]
            ax1.plot(subset['noise_level'].values, subset['Correct_Classification'].values, marker=markers[i], label=f'{fd_order:.0f}')
            ax1.set_title("Unique", fontsize=labelsize)
            ax1.set_xlabel('Noise Level', fontsize=labelsize)
            ax1.set_ylabel('Correctly Classified \n PDEs', fontsize=labelsize)
            ax1.set_xscale('log')
            #yticks for showing 1/2
            tick_positions = np.arange(0,N+1,1)  
            tick_labels = [f"{tick}/{N}" for tick in tick_positions]  
            ax1.set_yticks(tick_positions)
            ax1.set_yticklabels(tick_labels, fontsize=labelsize)
            #xticks for noise level
            ax1.tick_params(axis='x', labelsize=labelsize)
            ax1.tick_params(axis='y', labelsize=labelsize)
    #Plot non unique
    plot_class = plot_results[plot_results['True_Class'] == "non unique"]
    for i,fd_order in enumerate(fd_orders):
            N=N_nonunique
            subset = plot_class[plot_results['fd_order'] == fd_order]
            ax2.plot(subset['noise_level'].values, subset['Correct_Classification'].values, marker=markers[i], label=f'{fd_order:.0f}')
            ax2.set_title("Non unique", fontsize=labelsize)
            ax2.set_xlabel('Noise Level', fontsize=labelsize)
            ax2.set_xscale('log')
            ax2.legend(title='FD Order', fontsize=labelsize,title_fontsize=labelsize)
            tick_positions = np.arange(0,N+1,1)  
            tick_labels = [f"{tick}/{N}" for tick in tick_positions]  
            ax2.set_yticks(tick_positions) 
            ax2.set_yticklabels(tick_labels, fontsize=labelsize)
            ax2.tick_params(axis='x', labelsize=labelsize)
            ax2.tick_params(axis='y', labelsize=labelsize)
    if path!=None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def perform_experiment_jrc(noise_levels,fd_order,experiment_name,true_class,C2_param=1e-3,tickssize=16):
    results = pd.DataFrame(columns = ["True_Class", "Pred_Class","C2","noise_level","fd_order"])
    #Get data
    u,x,t,formula = experiment_data(n_samples=150,experiment_name=experiment_name)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    print(f"Performing experiment {experiment_name} {formula} with order {fd_order}, C2 = {C2_param:.2e}")
    subtitle=f"Experiment {experiment_name} {formula}, Order {fd_order}, C2_param = {C2_param:.2e}"
    T,X = np.meshgrid(t,x)
    C_upper_bounds_deriv = get_Cut_Cux_Cuxx_Cutx(formula,X,T,fd_order)
    
    for i,noise_level in enumerate(noise_levels):
        u_noise = add_noise(u,noise_level)
        eps = infinity_norm(u-u_noise)
        svs, lower_bounds,upper_bounds,space_range,time_range = get_results(u_noise,C_upper_bounds_deriv,fd_order,dt,dx,eps,C2_param)   
        upper_minus_svs = upper_bounds - svs
        lower_minus_svs = lower_bounds - svs
        
        # Find out the classifcation
        # If the Jacobian has full rank for at least one data point →PDE unique
        if ((lower_minus_svs < 0) & (upper_minus_svs < 0)).any() :
            predicted_class = "unique"
        # If the PDE is algebraic and at every data point the Jacobian does not have full rank
        # -> PDE non unique
        elif ((upper_minus_svs > 0) & (lower_minus_svs > 0)).all() :
            predicted_class = "non unique"
        else:
            predicted_class = None
        results.loc[len(results)] = [true_class, predicted_class,C2_param,noise_level,fd_order]
    results["Correct_Classification"]=results["True_Class"]==results["Pred_Class"]
    return results    


def perform_experiment_sfranco(noise_levels,experiment_name,true_class):
    #Get data
    u,x,t,formula = experiment_data(n_samples=150,experiment_name=experiment_name)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    print(f"Performing Experiment {experiment_name}")
    result = get_result_df(u,dx,noise_levels,orders=range(2,14,2),boundary=True)
    result.drop(["threshold_approx_nonuniq","threshold_approx_uniq"],axis=1)
    result["True_Class"]=true_class
    result['Pred_Class']=np.nan
    #if svs <= nonuniq and svs < uniq -> pde non unique
    condition = (result["ratio"] <= result["threshold_exact_nonuniq"]) & (result["ratio"] < result["threshold_exact_uniq"])
    result.loc[condition, 'Pred_Class'] = "non unique"
    # if svs >= uniq and svs > uniq -> pde unique
    condition = (result["ratio"]>=result["threshold_exact_uniq"]) & (result["ratio"]>result["threshold_exact_nonuniq"])
    result.loc[condition, 'Pred_Class'] = "unique"
    # Get if it is correctly classified
    result["Correct_Classification"]=result["True_Class"]==result["Pred_Class"]
    return result