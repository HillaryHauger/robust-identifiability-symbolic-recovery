from error_bounds import *
import numpy as np
import pysindy as ps
from test_data import *
import sympy
from numpy.linalg import svd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd

"""
This function computes the derivatives needed for the jacobian:
drop_endpoints: specifies if FiniteDifference should use the endpoints for calculation or not
remove_endpoints: if drop_endpoints=True remove_endpoints removes all the NaN values in the derivatives
such that all derivatives have the same shape
"""
def get_derivatives(u,dt,dx,fd_order,drop_endpoints=True,remove_endpoints=True):
    #Compute the derivatives
    ut_fd = ps.FiniteDifference(order=fd_order, d=1, axis=1, drop_endpoints=drop_endpoints)._differentiate(u, t=dt)
    utx_fd = ps.FiniteDifference(order=fd_order, d=1, axis=0, drop_endpoints=drop_endpoints)._differentiate(ut_fd, dx)
    ux_fd = ps.FiniteDifference(order=fd_order, d=1, axis=0, drop_endpoints=drop_endpoints)._differentiate(u, dx)
    uxx_fd = ps.FiniteDifference(order=fd_order, d=2, axis=0, drop_endpoints=drop_endpoints)._differentiate(u, dx) 

    if remove_endpoints:
        #Filter out the boundary values
        filter_func=utx_fd
        ut_fd= ut_fd[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
        utx_fd= utx_fd[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
        ux_fd= ux_fd[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
        uxx_fd= uxx_fd[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
    return ut_fd, utx_fd,ux_fd, uxx_fd

"""
This function filters all NaN values specified in filter_func out of func_fd.
func_fd: the function calculate with finite differences
func_true: the true derivative
filter_func: the function we want to filter out the nan values with
"""
def drop_endpoints(func_fd,func_true,filter_func):
    func_true = func_true[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
    func_fd= func_fd[~np.isnan(filter_func).all(axis=1)][:, ~np.isnan(filter_func).all(axis=0)]
    return func_fd,func_true

"""
This function calculates the upper error bound for the jacobian of the function g =(u | ux)
"""
def get_upper_bound_jacobian(eps,fd_order,Cut,Cux,Cuxx,Cutx,dt,dx,Cxi=1.0):

    bound_ut = upper_bound_central_differences(eps,fd_order,order_derivative=1,Cu=Cut,Cxi=Cxi,h=dt)
    bound_ux = upper_bound_central_differences(eps,fd_order,order_derivative=1,Cu=Cux,Cxi=Cxi,h=dx)
    bound_uxx = upper_bound_central_differences(eps,fd_order,order_derivative=2,Cu=Cuxx,Cxi=Cxi,h=dx)
    bound_utx = upper_bound_central_differences(bound_ut,fd_order,order_derivative=1,Cu=Cutx,Cxi=Cxi,h=dx)

    upper_bound = np.sqrt(bound_ut**2+bound_ux**2 + bound_uxx**2+bound_utx**2)
    
    return upper_bound

"""
This function calculates the upper bounds for the derivative u fd_order+deriv_order
for ut, ux, uxx and utx. 
These bounds are needed for estimating the error of the Jacobian.
"""
def get_Cut_Cux_Cuxx_Cutx(formula,X,T,fd_order):
    Cut = get_upper_bound_all_deriv_up_tofd_order(formula,'t', X,T,fd_order,deriv_order=1)
    Cux = get_upper_bound_all_deriv_up_tofd_order(formula,'x', X,T,fd_order,deriv_order=1)
    Cuxx = get_upper_bound_all_deriv_up_tofd_order(formula,'x', X,T,fd_order,deriv_order=2)
    Cutx = get_upper_bound_all_deriv_up_tofd_order(sympy.diff(formula,sympy.Symbol('t'),1),'x',X,T,fd_order,deriv_order=1)
    return (Cut, Cux, Cuxx, Cutx)

"""
This function calculates the singular values, lower and upper bounds for a function u
and an fd_order for different datapoints speficied in space and time range
u: function u
C_upper_bounds: These are the bounds for the fd_order + deriv_order of u calculated in get_Cut_Cux_Cuxx_Cutx
fd_order: finite differences order
eps: ps is the upper bound on the noise on u |u-u_noise|_infty < eps
C2_param: specifies lower bound for on > C2 > 0 with C2=max(C2_param*sv_max,sv_min*0.5)
return: svs - singular values calculated with finite differences at specified data points
        lower_bounds/upper_bounds: for identifying wether a PDE is unique or not
        space_range/time_range: data_points where the function is evaluated
"""
def get_results(u,C_upper_bounds_deriv,fd_order,dt,dx,eps,C2_param=1e-4):
    #Upper bounds on fd_order+deriv_order derivative of u
    Cut, Cux, Cuxx, Cutx= C_upper_bounds_deriv
    #Calculate finite differences
    ut_fd, utx_fd,ux_fd, uxx_fd= get_derivatives(u,dt,dx,fd_order)
    #Data points
    space_range = int(utx_fd.shape[0]/10-1)
    time_range = int(utx_fd.shape[1]/10-1)
    #For saving results
    svs = np.zeros([space_range, time_range])
    upper_bounds = np.zeros([space_range, time_range])
    lower_bounds = np.zeros([space_range, time_range])
    
    for i in range(space_range):
        for j in range(time_range):
            x_i, t_j = i * 10 + 10, j * 10 + 10
            jacobian_fd = np.array([[ut_fd[x_i,t_j], ux_fd[x_i,t_j]], [utx_fd[x_i,t_j], uxx_fd[x_i,t_j]]]).reshape(2,2)

            sv_fd = svd(jacobian_fd, compute_uv=False)
            sv_min =sv_fd[-1]
            sv_max =sv_fd[0]
    
            C1=sv_max*1.5
            C=sv_max*0.5
            C2=max(C2_param*sv_max,sv_min*0.5)
            #assert(C2<=C1)
            #assert(C2<=C)

            upper_bound_jacobian = get_upper_bound_jacobian(eps,fd_order,Cut,Cux,Cuxx,Cutx,dt,dx)
            #print(upper_bound_jacobian)
            lower_bound = lower_bound_nonsingular_matrix(C1,C2,upper_bound_jacobian)
            upper_bound = upper_bound_singular_matrix(C,upper_bound_jacobian)

            #Save the results 
            svs[i,j] = sv_min/sv_max
            lower_bounds[i,j]=lower_bound
            upper_bounds[i,j]=upper_bound
    return svs, lower_bounds,upper_bounds,space_range,time_range
"""
This function performs one experiment for an experiment_name.
It fethches the data and calculates the svs,lower_bounds,upper_bounds for all noise_levels.
Additionally it plots the results.
"""
def perform_experiment(noise_levels,fd_order,experiment_name,C2_param=1e-3,tickssize=16):
    
    #Get data,
    u,x,t,formula = experiment_data(n_samples=150,experiment_name=experiment_name)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    print(f"Performing experiment {experiment_name} {formula} with order {fd_order}, C2 = {C2_param:.2e}")
    subtitle=f"Experiment {experiment_name} {formula}, Order {fd_order}, C2_param = {C2_param:.2e}"
    T,X = np.meshgrid(t,x)
    C_upper_bounds_deriv = get_Cut_Cux_Cuxx_Cutx(formula,X,T,fd_order)

    
    fig, axes = plt.subplots(2,6, figsize=(24,8))
    
    for i,noise_level in enumerate(noise_levels):
        u_noise = add_noise(u,noise_level)
        eps = infinity_norm(u-u_noise)
        
        svs, lower_bounds,upper_bounds,space_range,time_range = get_results(u_noise,C_upper_bounds_deriv,fd_order,dt,dx,eps,C2_param)   
        t_grid, x_grid = (np.arange(time_range) * 10 + 10) / len(t) * (t[len(t)-1] - t[0]) + t[0], (np.arange(space_range) * 10 + 10) / len(x) * (x[len(x)-1] - x[0]) + x[0]
     
        #Plot the ratios
    
        c = axes[i//2,i*3%6+0].pcolor(t_grid, x_grid, svs)
        axes[i//2,i*3%6+0].set_title(r"$\frac{\sigma_n}{\sigma_1}$", fontsize=tickssize)
        axes[i//2,i*3%6+0].set_ylabel('x', fontsize=tickssize)
        axes[i//2,i*3%6+0].set_xlabel('t', fontsize=tickssize)
        fig.colorbar(c, ax=axes[i//2,i*3%6])
        
        upper_minus_svs = upper_bounds - svs
        lower_minus_svs = lower_bounds -svs
        
        max_lower_svs= np.max(np.abs(lower_minus_svs))
        max_upper_svs = np.max(np.abs(upper_minus_svs))
        #max_svs = max(max_lower_svs, max_upper_svs)
    
        #Plot difference to upper bound
        max_svs = max_upper_svs
        plt.suptitle(subtitle, fontsize=tickssize,y=1.1)
        axes[i//2,i*3%6+1].set_title(f'Noise level {noise_level} \n Non Unique', fontsize=tickssize)
        c=axes[i//2,i*3%6+1].pcolor(t_grid, x_grid, upper_minus_svs,cmap=cmap_red_green,vmin=-max_svs, vmax=max_svs)
        axes[i//2,i*3%6+1].set_yticks([])
        fig.colorbar(c, ax=axes[i//2,i*3%6+1])
        
       
        #Plot difference to lower bound
        max_svs = max_lower_svs
        axes[i//2,i*3%6+2].set_title('Unique', fontsize=tickssize)
        c = axes[i//2,i*3%6+2].pcolor(t_grid, x_grid, lower_minus_svs, cmap=cmap_green_red, vmin=-max_svs, vmax=max_svs)
        axes[i//2,i*3%6+2].set_yticks([])
        fig.colorbar(c, ax=axes[i//2,i*3%6+2])

        #Plot legend for True and False
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='True'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='False')
        ]

        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize='large')

        # Adjust subplot layout to make room for the legends
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()



#####################################
###########      Plots    ###########
#####################################
#Defining colors for the plots: green = true , red = false
green=(100/255,250/255,100/255)#(0,1,0)
neutral_color = (1, 1, 1)  
red=(241/255,13/255,30/255)
cmap_green_red = LinearSegmentedColormap.from_list('RedGreen', [green,neutral_color,red], N=256)
cmap_red_green = LinearSegmentedColormap.from_list('GreenRed', [red,neutral_color,green], N=256)

"""
This function plots the results gotten from get_results
"""
def plot_results(upper_bounds, lower_bounds,svs,time_range,space_range,t, x,subtitle="",subtitle_size=16,tickssize=16):
    t_grid, x_grid = (np.arange(time_range) * 10 + 10) / len(t) * (t[len(t)-1] - t[0]) + t[0], (np.arange(space_range) * 10 + 10) / len(x) * (x[len(x)-1] - x[0]) + x[0]
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    #Plot the ratios
    c = axes[0].pcolor(t_grid, x_grid, svs)
    axes[0].set_title(r"$\frac{\sigma_n}{\sigma_1}$", fontsize=tickssize)
    axes[0].set_ylabel('x', fontsize=tickssize)
    axes[0].set_xlabel('t', fontsize=tickssize)
    fig.colorbar(c, ax=axes[0])
    
    upper_minus_svs = upper_bounds - svs # <= 0 if unique >= 0 if non-unique
    lower_minus_svs = lower_bounds - svs # <= 0 if unique >= 0 if non-unique
    max_lower_svs= np.max(np.abs(lower_minus_svs))
    max_upper_svs = np.max(np.abs(upper_minus_svs))
    max_svs = max(max_lower_svs, max_upper_svs)
    
    #Plot difference to upper bound
    plt.suptitle(subtitle, fontsize=subtitle_size,y=1.1)
    axes[1].set_title('Non Unique', fontsize=tickssize)
    c=axes[1].pcolor(t_grid, x_grid, upper_minus_svs,cmap=cmap_red_green,vmin=-max_svs, vmax=max_svs)
    axes[1].set_yticks
    fig.colorbar(c, ax=axes[1])
    
    #Plot difference to lower bound
    
    axes[2].set_title('Unique', fontsize=tickssize)
    c = axes[2].pcolor(t_grid, x_grid, lower_minus_svs, cmap=cmap_green_red, vmin=-max_svs, vmax=max_svs)
    axes[2].set_yticks([])
    fig.colorbar(c, ax=axes[2])

    #Plot legend for True and False
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='True'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='False')
    ]

    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize='large')

    # Adjust subplot layout to make room for the legends
    plt.subplots_adjust(bottom=0.2)
    #plt.savefig(f"Images/JRC_{subtitle}")
    plt.tight_layout()


def plot_ratio_upper_lower_old(svs,lower_bounds,upper_bounds,title=""):
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
    n=len(svs.flatten())
    ax.set_title(title)

    max_value = np.maximum.reduce([svs.flatten().max(), lower_bounds.flatten().max(), upper_bounds.flatten().max()])

    ax.plot( svs.flatten(),color='blue')
    ax.fill_between(np.arange(n),lower_bounds.flatten(), max_value,color='coral', alpha=0.3)
    ax.plot(lower_bounds.flatten(),label="unique",color='coral')
    ax.set_ylabel('Threshold/Ratio')


    ax.fill_between(np.arange(n),upper_bounds.flatten(), y2=0, color='aquamarine', alpha=0.3)
    ax.set_yscale('log')
    ax.plot(upper_bounds.flatten(),label="not unique", color='aquamarine')

    ax.set_xlabel('Datapoints')
    ax.legend(loc=4)

######################################
##### C2 Parameter Analysis  #########
######################################
def get_results_C2(C2_params,noise_levels,fd_order,experiment_name,true_class):
    results = pd.DataFrame(columns = ["True_Class", "Pred_Class","C2","noise_level","fd_order"])
    #Get data
    u,x,t,formula = experiment_data(n_samples=150,experiment_name=experiment_name)
    T,X = np.meshgrid(t,x)
    C_upper_bounds_deriv = get_Cut_Cux_Cuxx_Cutx(formula,X,T,fd_order)
    dx=x[1]-x[0]
    dt=t[1]-t[0]
    print(f"Performing experiment {experiment_name} {formula} with order {fd_order}")
    
    for C2_param in C2_params:
        print(f"C2 = max({C2_param:.3e} * sv_max,0.5*sv_min)")          
        for i,noise_level in enumerate(noise_levels):
            u_noise = add_noise(u,noise_level)
            eps = infinity_norm(u-u_noise)
            
            svs, lower_bounds,upper_bounds,space_range,time_range =  get_results(u_noise,C_upper_bounds_deriv,fd_order,dt,dx,eps,C2_param)
           
            upper_minus_svs = upper_bounds - svs
            lower_minus_svs = lower_bounds - svs

            
            # Find out the classifcation
            # If the Jacobian has full rank for at least one data point →PDE unique
            if ((lower_minus_svs < 0) & (upper_minus_svs < 0)).any() :
                predicted_class = "unique"
            # If the PDE is algebraic and at every data point the Jacobian does not have full rank
            # -> PDE non unique
            elif ((upper_minus_svs > 0) & (lower_minus_svs > 0)).all() :
                #print("Prediction is non unique")
                #print("upper", upper_minus_svs)
                #print("lower", lower_minus_svs)
                predicted_class = "non unique"
            else:
                #print("upper_minus", upper_minus_svs)
                #print("lower_minus", lower_minus_svs)
                predicted_class = None
    
            results.loc[len(results)] = [true_class, predicted_class,C2_param,noise_level,fd_order]
    results["Correct_Classification"]=results["True_Class"]==results["Pred_Class"]
    return results

def get_df_analysis(results):
    df_analysis = pd.DataFrame()
    df_analysis["Count_Correct_Class"] = results[["Correct_Classification","C2"]].groupby("C2").sum()
    df_analysis["Count_Total_Class"] = results[["True_Class","C2"]].groupby("C2").count()
    df_analysis["max_correct_noise_level"]= results[results['Correct_Classification']].groupby('C2')['noise_level'].max()
    df_analysis["Correct_Class_percentage"]=df_analysis["Count_Correct_Class"]/df_analysis["Count_Total_Class"]*100
    return df_analysis
    
#Check wether all classification before max_correct_noise_level where True
def check_classification_max_correct_noise_level(df_analysis,C2_params,results):
    for C2 in C2_params:
        df = df_analysis[df_analysis.index==C2]
        max_nl = df["max_correct_noise_level"].values[0]
        count_corr_cl = df["Count_Correct_Class"].values[0]
        count_check = len(results[results["noise_level"]<=max_nl].noise_level.unique())
        if count_corr_cl != count_check:
            print(f"For C2 = {C2} not all noise levels before {max_nl} were correctly classified. Please check the dataframe")
            print(f"Numer correct classified labels: {count_corr_cl}, Number it should be {count_check}")

def plot_results_C2(experiment_name,df_analysis,noise_levels):
    plt.title(f"{experiment_name}")
    plt.plot(df_analysis.index.values, df_analysis["Correct_Class_percentage"].values,marker = 'x',color='green',)
    plt.xlabel("C2")
    plt.ylabel("Percentage Correct Classification")
    tick_positions = np.arange(0, 110, 10)  # Generate tick positions from 0 to 100
    tick_labels = [f"{tick}% " for tick in tick_positions]  
    plt.yticks(tick_positions, tick_labels)
    plt.xscale("log")
    plt.show()

def plot_results_C2_noise_level(experiment_name,df_analysis,noise_levels):
    plt.title(f"{experiment_name}")
    plt.plot(df_analysis.index.values, df_analysis["max_correct_noise_level"].values,marker = 'x',color='green',)
    plt.xlabel("C2")
    plt.ylabel("Maximum correct classified noise_level")
    plt.yticks(noise_levels)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()