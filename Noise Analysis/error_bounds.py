import numpy as np
##############################################################
####   Functions for bounding finite differences error   #####
##############################################################
"""
Computes the coefficients of the derivitve of the kth Lagrangian Coefficients.
These are also the coefficients used for finite differences.
n: number of data points used for approximation/ order
k: number of lagrange coefficient
l: input to lagrange coefficient: x = x_l = x+ hl
"""
def lagrange_coefficient_derivative(n,k,l):
    erg=0.0        
    for j in range(n+1):
        if j!=k:
            tmp =1.0
            for i in range(n+1):
                  if i!=k and i!=j:
                    tmp *=(l-i)/(k-i)
                    #print(f"l-k = {l}-{i}")
            tmp*=1.0/(k-j)
            erg+=tmp
    return erg
    
"""
Sums up the above coefficients and is needed for calculating the measurment error
n: number of data points used for approximation/ order
l: input to lagrange coefficient: x = x_l = x+ hl
"""
def sum_lagrange_coefficient_derivative(n,l):
    erg=0.0
    for k in range(n+1):
        #print(f"L({n},{k},{l}) = {lagrange_coefficient_derivative(n,k,l)}")
        erg+= np.abs(lagrange_coefficient_derivative(n,k,l))
    return erg

"""
Approximation error for central differences: reduces with higher order
The approximation error is caused by finite differences itself.
"""
def appr_error_central_diff(order):
    assert(order%2==0) #Check if n is even
    bound = (np.math.factorial(int(order/2))**2)/np.math.factorial(order+1)
    return bound
    
# Approximation error backward differences
def appr_error_backward_diff(n):
    return 1/(n+1)

"""
Measurement error for central differences.
The measurement error is caused round off errors and does not decrease for higher order.
"""
def meas_error_central_diff(n):
    assert(n%2==0) #Check if n is even
    erg=sum_lagrange_coefficient_derivative(n,n/2)
    return erg
    
# Measurement error backward differences
def meas_error_backward_diff(n):
    erg=sum_lagrange_coefficient_derivative(n,0)
    return erg
    
"""
Error bound on the first derivative calculated with finite differences
eps: measurement error+round of error
h: dx for equispaced data
M: bound on (order+1)th derivative
"""
def error_bound_finite_diff(eps,h,M,order=2):
    eps+=np.finfo(float).eps # add machine precisoin
    if order%2==0: #even order
        C_app=appr_error_central_diff(order)
        C_meas=meas_error_central_diff(order)
    else: # odd order
        C_app=appr_error_backward_diff(order)
        C_meas=meas_error_backward_diff(order)
    #print(f" C_meas*eps/h + (h**order)*M*C_app = {C_meas:2.3e}*{eps:2.3e}/{h:2.3e} + {h**order:2.3e}*{M:2.3e}*{C_app:2.3e}")    
    return C_meas*eps/h + (h**order)*M*C_app
            
def infinity_norm(x):
    return np.max(np.abs(x))

##############################################################
####       Functions for bounding condition error        #####
##############################################################
"""
Computes error_bound  for ||g-g_noise||_F^2, where g=(u ux)
eps_two: ||u-u_noise||_2 two norm
eps_infty: ||u-u_noise||_infty infinity norm
h: dx for equispaced data
number_datapoints: number of total datapoints where u is evaluated
M: bound on nth derivative depending on the order
"""
def error_bound_g(eps_two,eps_infty,h,number_datapoints,M,order):
    E = eps_two**2 + number_datapoints*error_bound_finite_diff(eps_infty,h,M,order)**2
    return E

"""
Computes total error on condition o1/o2 without considering the supremum
E: error bound on ||g-g_noise||_F^2 calculated in above functions
sv_max: highest singular value on matrix without noise
sv_min: lowest singular value on matrix without noise
"""
def error_bound_condition(E,sv_max,sv_min):
    bound = np.sqrt(1/sv_min+(sv_max/sv_min**2)**2*E)
    return bound