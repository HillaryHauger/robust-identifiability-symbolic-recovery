import numpy as np

def infinity_norm(x):
    return np.max(np.abs(x))
    
#######################################################################################################
####   Functions for bounding finite differences error up to third derivative central differences #####
#######################################################################################################

"""
This calculates the derivative (up to third) of the residue r of the Lagrange polynomial.
It is needed for calculating the approximation error
Cu: We assume that all derivatives of u from order_fd+1 to order_fd+order_derivative can be bounded by Cu
Cxi: We assume that all derivatives of xi up to order_fd can be bounded by Cxi
h: h is the space between data points
order_derivative: is the order of the derivative
"""
def derivative_r(n,order_derivative,h,Cu,Cxi,m):

    if order_derivative>3:
        print(f"Derivative {order_derivative} is not implemented yet")
        return None
        
    result = 0.0
    pre = np.math.factorial(n + 1)

    #Calculate all products and sums necesseary for calculating the result
    if order_derivative>=1:
        product_term1=1.0
        for k in range(n + 1):
            if k != m:
                product_term1 *= (m - k)
        product_term1=abs(product_term1)

    if order_derivative  >=2:
        product_sum_term2 = 0.0
        for s in range(n + 1):
            if s!=m:
                product_term2 = 1.0
                for k in range(n + 1):
                    if k!=m and k!=s:
                        product_term2 *= (m - k)  
                product_sum_term2 += product_term2
        product_sum_term2=abs(product_sum_term2)
        
    if order_derivative >=3:
        product_sum_term3 = 0.0
  
        for s1 in range(n + 1):
                for s2 in range(s1+1,n + 1):
                        if s1!=m and s2!=m:
                            product_term3 = 1.0
                            for k in range(n + 1):
                                if k!=s1 and k!=s2 and k!=m and n!=2:
                                    product_term3 *= (m - k)
                            product_sum_term3 += 6*product_term3          
        product_sum_term3=abs(product_sum_term3)
        
        
    #Put everything together           
    if order_derivative==1:
        result = Cu*product_term1/pre*(h**n)
    elif order_derivative==2:
        result = Cxi*Cu*2*product_term1/pre*h**n + Cu*product_sum_term2/pre*h**(n-1)
    elif order_derivative==3:
        result = 3*((Cxi**2)*Cu+Cu*Cxi)*product_term1/pre*h**n + 3*Cu*Cxi*product_sum_term2/pre*h**(n-1) +Cu*product_sum_term3/pre*h**(n-2)
    return result

    

"""
Computes the coefficients of the derivative of the kth Lagrangian Coefficients.
These are also the coefficients used for finite differences.
It is needed for calculating the measurement error
n: number of data points used for approximation/ order
k: number of lagrange coefficient
l: input to lagrange coefficient: x = x_l = x+ hl
h: h is the space between data points
order_derivative: order of the derivative, can take values 1,2,3
"""

def derivative_Lnk(n,k,l,order_derivative,h):
    #Calculate preceeding product, same for all derivatives
    product_pre= 1.0
    for i in range(n + 1):
        if i != k:
            product_pre *= 1 / (k - i)        
    result=0.0 

    if order_derivative == 1: 
        result =1.0
        for i in range(n+1):
              if i!=k and i!=l and k!=l:
                result *=(l-i)     
        if l==k:
            result = 0.0

    elif order_derivative == 2:
        for s2 in range(n + 1):
            for s1 in range(s2+1,n + 1):
                product_1 = 1.0
                if s1!=k and s2!=k:
                    for i in range(n + 1):
                        if i != k and i != s1 and i != s2:
                            product_1 *= (l - i)
                else:
                    product_1 = 0.0
                result += 2*product_1


    elif order_derivative == 3:
        for s3 in range(n + 1):
            for s2 in range(s3+1,n + 1):
                for s1 in range(s2+1,n + 1):
                    if s1!=k and s2!=k and s3!=k:
                        product_1=1.0
                        for i in range(n + 1):
                            if i != k and i != s1 and i != s2 and i != s3:
                                product_1 *= (l - i)
                    else:
                        product_1=0.0
                    result += 6*product_1

    
    else:
        print(f"Derivative {order_derivative} is not implemented yet")
        result=None
    return result*product_pre/h**(order_derivative)

"""
This returns the measurement and approximation constants of calculating finite differences
"""
def get_measurement_approximation_error(order_fd, order_derivative,m,h,Cu,Cxi):
     #Get measurement error
    C_meas=0.0
    for k in range(order_fd+1):
            lagrange_der=derivative_Lnk(order_fd,k,m,order_derivative,h)
            C_meas += abs(lagrange_der)
    #Calculate approximation error
    C_app  = derivative_r(order_fd,order_derivative,h,Cu,Cxi,m)
    return C_meas,C_app
    
"""
This calcuates an upper bound on the error of central differences.
Cu: We assume that all derivatives of u up order_fd+order_derivative can be bounded by Cu
Cxi: We assume that all derivatives of xi up to order_fd can be bounded by Cxi
h: h is the space between data points
eps: eps is the upper bound on the noise on u |u-u_noise|_infty < eps
order_fd: is the finite difference_order
order_derivative: is the order of the derivative
"""
def upper_bound_central_differences(eps,order_fd,order_derivative,Cu,Cxi,h):
    if order_derivative>3:
        print(f"Derivative {order_derivative} is not implemented yet")
        return None
    if order_fd%2!=0:
        print(f"Only central differences is implemented, finite differences order {order_fd} can not be calculated")
        return None
    m=int(order_fd/2)
    C_meas,C_app = get_measurement_approximation_error(order_fd,order_derivative,m,h,Cu,Cxi)
    upper_bound = C_app+eps*C_meas
    return upper_bound

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
def error_bound_g(eps_two,eps_infty,h,number_datapoints,order_fd, max_order_derivative=1,Cu=1.0,Cxi=1.0):
    E = eps_two**2
    #Add error bound for each order
    for order_derivative in range(1,max_order_derivative+1):
       # print(order_derivative)
        E+=number_datapoints*upper_bound_central_differences(eps_infty,order_fd,order_derivative,Cu,Cxi,h)**2
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


##############################################################
#### Functions for bounding reversed condition error     #####
##############################################################

"""
This function should upper bound the reversed condition for singular matrices 
C1:lower bound of biggest singular value: C <= o_max
eps: lower obund for frobenius norm of error matrix |E|_F <= eps
"""
def upper_bound_singular_matrix(C,eps):
    if C-eps<=0:
        #print(f"Error is too big C<eps with C = {C:.3e}, eps = {eps:.3e}: no upper bound can be calculated")
        return 1.0
        
    bound= eps/(C-eps)
    bound = min(bound, 1.0) # o_min/o_max <=1 in all cases
    #print(f"UB: {bound:.3e} = {eps:.3e}/({C:.3e}-{eps:.3e})")
    return bound

"""
This function should lower bound the reversed condition for nonsingular matrices 
where on>C2>0
C1: upper bound for o1 < C1
C2: lower bound for on > C2 > 0
eps: lower obund for frobenius norm of error matrix |E|_F <= eps
The question is how to choose C2??
"""
def lower_bound_nonsingular_matrix(C1,C2,eps):
    bound= (C2-eps)/(C1+eps)
    bound = max(bound,0.0)
    #print(f"LB: {bound:.3e} = ({C2:.3e}-{eps:.3e})/({C1:.3e}+{eps:.3e})")
    return bound
    
"""Threshold if values are beneath classify as non unique PDE
E: upper bound on the frobenius  |g-g_noise|_frobenius <=E where g =(u|u_x|...)
C: lower bound of biggest singular value: C <= o_max
"""
def calc_threshold_nonuniq(E,C):
    T = upper_bound_singular_matrix(C,E)
    return T
    
"""Threshold if values are above classify as unique PDE
E: upper bound on the frobenius  |g-g_noise|_frobenius <=E where g =(u|u_x|...)
C1: upper bound of biggest singular value: C1 => o_max
C2: upper bound of biggest singular value: C2 <= o_max
"""
def calc_threshold_uniq(C1,C2,E):
    T = lower_bound_nonsingular_matrix(C1,C2,E)
    return T