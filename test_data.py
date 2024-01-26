#This file creates data in order to test the algorithms on how
#well they identify uniqueness

import numpy as np
import sympy

def create_data_2d(T_start=0, T_end=5, L_x_start=-5,L_x_end=5, N_t=200, N_x=200):
    t = np.linspace(T_start, T_end, num=N_t)
    x = np.linspace(L_x_start,L_x_end, num=N_x)
    T,X = np.meshgrid(t,x)
    return T,X,t,x

def create_data_3d(T_start, T_end, L_x,L_y, N_t, N_x,N_y):
    t = np.linspace(T_start, T_end, num=N_t)
    x = np.linspace(-L_x/2.0, L_x/2.0, num=N_x)
    y = np.linspace(-L_y/2.0, L_y/2.0, num=N_y)
    T,X,Y = np.meshgrid(t,x,y)
    return T,X,Y,t,x,y

# adds noise to data dependent on norm
def add_noise(u,target_noise,seed=1234):
    np.random.seed(seed)
    var = target_noise * np.sqrt(np.mean(np.square(u)))
    u_noise = u + np.random.normal(0, var, size=u.shape)
    return u_noise

# normalisation especially for jacobi matrix: in order to get conclusive singular values
# uses the frobenius norm to normalise
def normalise_frobenius(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm
    return matrix

# normalisation especially for jacobi matrix: in order to get conclusive singular values
def normalise_minmax(matrix):
    max=matrix.max()
    matrix = matrix/max
    return matrix

def get_experiment_names():
    names = ['linear_nonunique_1','linear_nonunique_1.1','linear_nonunique_1.2','linear_unique_1','linear_unique_2','algebraic_nonunique_1',
    'algebraic_nonunique_kdv', 'analytic_unique_1', 'analytic_unique_2','analytic_nonunique_1']
    exp_name_dict = {}
    for name in names:
        u,x,t,formula= experiment_data(1, name)
        exp_name_dict[name] = formula
    return exp_name_dict

def experiment_data(n_samples, experiment_name):
    
    if experiment_name == 'linear_nonunique_1':
        T,X,t,x = create_data_2d(N_t=n_samples,N_x=n_samples)
        a= np.random.randn()
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = module.exp(X-a*T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'linear_unique_1':
        T,X,t,x = create_data_2d(N_t=n_samples,N_x=n_samples)
        a,b= np.random.randn(2)
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = (X+b*T)*module.exp(a*T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'linear_nonunique_1.1':
        T,X,t,x = create_data_2d(N_t=n_samples,N_x=n_samples)
        a= np.random.randn()
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = module.cos(X-a*T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)   

    elif experiment_name == 'linear_nonunique_1.2':
        T,X,t,x = create_data_2d(N_t=n_samples,N_x=n_samples)
        a= np.random.randn()
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = module.sin(X-a*T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)         

    elif experiment_name == 'linear_unique_2':
        T,X,t,x = create_data_2d(N_t=n_samples,N_x=n_samples)
        c= np.random.randn()
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = module.sin(X-c*T) + module.sin(X+c*T) +4*X*T
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'algebraic_nonunique_1':
        T,X,t,x = create_data_2d(T_start=1, T_end=5, L_x_start=1,L_x_end=5,N_t=n_samples,N_x=n_samples)
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = 1/(X+T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'algebraic_nonunique_kdv':
        T,X,t,x = create_data_2d(T_start=1, T_end=5, L_x_start=1,L_x_end=5,N_t=n_samples,N_x=n_samples)
        c = np.abs(np.random.randn())
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = (c/2)*module.cosh(np.sqrt(c)/2*(X-c*T))**(-2)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)
    
    elif experiment_name == 'analytic_unique_1':
        T,X,t,x = create_data_2d(T_start=1, T_end=5, L_x_start=1,L_x_end=5,N_t=n_samples,N_x=n_samples)
        a = np.abs(np.random.randn())
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
                f = (X + T)*module.acos(1/module.cosh(a*T))
            else:
                f = (X + T)*module.arccos(1/module.cosh(a*T))
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'analytic_unique_2':
        T,X,t,x = create_data_2d(T_start=1, T_end=5, L_x_start=1,L_x_end=5,N_t=n_samples,N_x=n_samples)
        a = np.abs(np.random.randn())
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
                f = (X + T)*module.asin(1/module.cosh(a*T))
            else:
                f = (X + T)*module.arcsin(1/module.cosh(a*T))
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    elif experiment_name == 'analytic_nonunique_1':
        T,X,t,x = create_data_2d(T_start=0, T_end=5, L_x_start=0,L_x_end=5,N_t=n_samples,N_x=n_samples)
        a = np.abs(np.random.randn())
        def func(T,X, module): 
            if module == sympy:
                T = sympy.Symbol('t')
                X= sympy.Symbol('x')
            f = module.sin(X+T)
            return f
        u = func(T,X,np) 
        formula = func(T,X,sympy)

    else:
        raise NotImplementedError(f'Experiment {experiment_name} not implemented yet.')
        
    return u,x,t,formula

