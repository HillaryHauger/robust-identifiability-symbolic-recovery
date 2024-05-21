from error_bounds import *
import numpy as np
import pysindy as ps

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
    Cut, Cux, Cu, Cutx= C_upper_bounds_deriv
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

            upper_bound_jacobian = get_upper_bound_jacobian(eps,fd_order,Cut,Cux,Cuxx,Cutx,dt,dx)
            lower_bound = lower_bound_nonsingular_matrix(C1,C2,upper_bound_jacobian)
            upper_bound = upper_bound_singular_matrix(C,upper_bound_jacobian)

            #Save the results 
            svs[i,j] = sv_min/sv_max
            lower_bounds[i,j]=lower_bound
            upper_bounds[i,j]=upper_bound
    return svs, lower_bounds,upper_bounds,space_range,time_range


#####################################
###########      Plots    ###########
#####################################

"""
This function plots the results gotten from get_results
"""
def plot_results(upper_bounds, lower_bounds,svs,time_range,space_range,subtitle="",subtitle_size=16,tickssize=16):
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
    max_svs= np.max(np.abs(upper_minus_svs))
    #Plot difference to upper bound
    plt.suptitle(subtitle, fontsize=subtitle_size,y=1.1)
    
    axes[1].set_title('Non Unique', fontsize=tickssize)
    c=axes[1].pcolor(t_grid, x_grid, upper_minus_svs,cmap=cmap_red_green,vmin=-max_svs, vmax=max_svs)
    axes[1].set_yticks
    fig.colorbar(c, ax=axes[1])
    
    #Plot difference to lower bound
    max_svs= np.max(np.abs(lower_minus_svs))
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