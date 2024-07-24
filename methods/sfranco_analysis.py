import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import pandas as pd
import pysindy as ps
from utils.test_data import *
from methods.error_bounds import *

#Plots classification for different noise leevels
def plot_uniq_non_uniq_class_with_noise_levels(noise_levels,result,image_path=None,title='Classification for different Noise Levels'):
    num_cols = 4
    num_rows = int(np.ceil(len(noise_levels)/num_cols))
    fig, axs = plt.subplots(num_rows, num_cols,  figsize=(5*num_cols, 4 *num_rows))
    fig.suptitle(title, fontsize=16)
    
    # Iterate over noise levels
    j=0
    for i, noise_level in enumerate(noise_levels,start=0):
        df = result[result["noise_level"] == noise_level]
  
        max_val = max(result["ratio"].max(),df["threshold_exact_uniq"].max(),df["threshold_exact_nonuniq"].max()) 
        axs[i//(num_rows+1), j%(num_cols)].set_title(f"Noise Level {noise_level}")
        axs[i//(num_rows+1), j%(num_cols)].plot(df["ratio"], color='blue',label =r"$\rho(\tilde{G})$")
        axs[i//(num_rows+1), j%(num_cols)].plot(df["threshold_exact_nonuniq"], ':', label="non unique", color='aquamarine')
        axs[i//(num_rows+1), j%(num_cols)].fill_between(df.index, 0, df["threshold_exact_nonuniq"], color='aquamarine', alpha=0.3)
        
        axs[i//(num_rows+1), (j)%num_cols].plot(df["threshold_exact_uniq"], ':', label="unique", color='coral')
        axs[i//(num_rows+1), (j)%num_cols].fill_between(df.index, df["threshold_exact_uniq"], y2=max_val, color='coral', alpha=0.3)
        axs[i//(num_rows+1), (j)%num_cols].set_yscale('log')
        
        #Only show legend in first plot
        if i ==0:
            axs[i//(num_rows+1), j%(num_cols)].legend(loc=4)
            axs[i//(num_rows+1), j%(num_cols)].legend(loc=4)
        #Only show y and x label in last plot
        
        if i//(num_rows+1) == num_rows-1:
            axs[i//(num_rows+1), j%(num_cols)].set_xticks(df.index)
            axs[i//(num_rows+1), j%(num_cols)].set_xticklabels(df.order)
            if j%(num_cols) == 0: 
                axs[i//(num_rows+1), j%(num_cols)].set_ylabel('Threshold/Ratio')
                axs[i//(num_rows+1), j%(num_cols)].set_xlabel('Order')
        else:
            #Get rid of x labels
            axs[i//(num_rows+1), j%(num_cols)].set_xticks([])
            axs[i//(num_rows+1), j%(num_cols)].set_xticks([])
        
            
        j+=1
    
    # Show the plot
    #fig.savefig(image_path)
    plt.show()

def classify_string(linear_string):
    if 'nonunique' in linear_string:
        return 'nonunique'
    elif 'unique' in linear_string:
        return 'unique'
    else:
        return None


"""
This calculates g for g = (u,u_x,u_xx,...) up to max_derivative
for the specified finite differences order
boundary: specifies if the derivatives drop the endpoints or not
"""
def calculate_g_derivative(u,dx,order,max_derivative=1, boundary=False):
    u_flat = u.flatten()
    N=len(u_flat)

    #Append u to u _derivative
    if boundary:
        #For boundary True calculate mask of maximum derivative:
        u_deriv_order = ps.FiniteDifference(order=order,d=max_derivative, axis=0, drop_endpoints=True)._differentiate(u, dx)
        row_mask = ~np.isnan(u_deriv_order).all(axis=1)
        col_mask = ~np.isnan(u_deriv_order).all(axis=0)
        u_deriv = [u[row_mask][:, col_mask].flatten().reshape(-1,1)]
    else:
        u_deriv = [u_flat.reshape(N,1)]
        
    #Append the other derivatives
    for derivative_order in range(1,max_derivative+1):
        if boundary:
            u_deriv_order = ps.FiniteDifference(order=order,d=derivative_order, axis=0, drop_endpoints=True)._differentiate(u, dx)
            u_deriv_order=u_deriv_order[row_mask][:, col_mask]
            u_deriv_order_flat = u_deriv_order.flatten()
            u_deriv.append(u_deriv_order_flat.reshape(-1,1))
            
        else:
            u_deriv_order = ps.FiniteDifference(order=order,d=derivative_order, axis=0, drop_endpoints=False)._differentiate(u, dx)
            u_deriv_order_flat = u_deriv_order.flatten()
            u_deriv.append(u_deriv_order_flat.reshape(N,1)) 
            
    g = np.concatenate(u_deriv, axis=1)
    return g 

"""
This function goes through all wanted noise levels and calculates the thresholds,
ratios,... and saves everything in a dataframe result
"""
def get_result_df(u,dx,noise_levels,orders=range(2,10,2),max_order_derivative=1,Cxi=1.0,C2_param=1e-4, boundary=False):
    result= pd.DataFrame(columns=["noise_level","order","ratio","threshold_approx_uniq","threshold_exact_uniq","threshold_approx_nonuniq","threshold_exact_nonuniq","sv_max","sv_min","C","C1","C2"])
    M=infinity_norm(u)
    true_g = calculate_g_derivative(u,dx,2,max_order_derivative)
    
    unorm2=np.sqrt(np.mean(np.square(u)))
    
    for noise_level in noise_levels:    
        #Add noise 
        var = noise_level * unorm2
        noise = np.random.normal(0, var, size=u.shape)
        u_noise = u + noise
        eps_two = np.linalg.norm(u-u_noise)
        eps_infty = infinity_norm(u-u_noise)
    
        # Need intitial value for C,C1 and C2
        g_noise = calculate_g_derivative(u_noise,dx,2,max_order_derivative,boundary)
        sv = svd(g_noise, compute_uv=False)
        #Choose C,C1,C2
         # For uniqueness classification
        C=sv[0]*0.5 #<=sv_max
        # For non-uniqueness classification
        C1=sv[0]*1.5 #>=sv_max
        C2=max(C2_param*sv[0],sv[-1]*0.5)#<=sv_min
        
        for order in orders:
            #Perform svd and finitedifference to get ratio
            g_noise = calculate_g_derivative(u_noise,dx,order,max_order_derivative,boundary)
            sv = svd(g_noise, compute_uv=False)
            ratio=sv[-1]/sv[0]
    
            #Potential upper bounds for |G-G_noise| 
            E1=np.sqrt(eps_infty/dx+dx**order)
            E2=np.sqrt( error_bound_g(eps_two,eps_infty,dx,u.shape[0],order, max_order_derivative=max_order_derivative,Cu=M,Cxi=Cxi))
        

            #Calculate thresholds
            threshold_approx_uniq =  calc_threshold_uniq(C1,C2,E1)
            threshold_exact_uniq =  calc_threshold_uniq(C1,C2,E2)
      
            threshold_approx_nonuniq = calc_threshold_nonuniq(E1,C)
            threshold_exact_nonuniq = calc_threshold_nonuniq(E2,C)
            
            #Save results in dataframe
            result.loc[len(result.index)] = [noise_level,order,ratio,threshold_approx_uniq,threshold_exact_uniq,threshold_approx_nonuniq,threshold_exact_nonuniq,sv[0],sv[-1],C,C1,C2]

    return result

def perform_experiment(experiment_name,n_samples=50,noise_levels = [0]+[10**(-10+i) for i in range(0,9)],orders=range(2,10,2),max_order_derivative=1, Cxi=1.0,save_result = False):
    #Get data
    u,x,t,formula = experiment_data(n_samples=n_samples,experiment_name=experiment_name)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    print("Performing experiment ",experiment_name, "with formula", formula)
    result = get_result_df(u,dx,noise_levels,max_order_derivative=max_order_derivative,Cxi=Cxi,boundary=True)
    #Save results
    # Creating the directory if it doesn't exist
    if save_result:
        directory_path="../results/" + experiment_name
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        result.to_csv(directory_path+f"/results_nr_franco_orderderivative{max_order_derivative}.csv")
    return result
