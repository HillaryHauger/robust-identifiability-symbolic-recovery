#!/usr/bin/env python
# coding: utf-8

# # Implementation of Frequency Methods
# Following functions are implemented:
# - Calulate derivative with FFT
# - Calculate extension of a function(one and multidimensional)
# - Calculate derivative with FFT and extension of a function


# Packages
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg
import itertools
import pysindy as ps

from scipy.optimize import minimize
from scipy.fft import fftn, ifftn, fftfreq,fft


# ### Functions for calculating derivative and extension

# Calculates the spectral derivative for n-dimensional input and order
# Takes ax and d as liste e.g. ax = [0], d =[0.1]
def calc_deriv_fftn(f,ax,d):
    if isinstance(ax, list) == False or isinstance(ax, list) == False:
        print("ax and d are only valid as a list")
        print("Exit calc_deriv_fftn")
        return None
    order = len(ax)
    fftf = fftn(f)
    frequencies = []
    # Calculate frequencies
    for i in range(order):
        frequencies.append(fftfreq(f.shape[ax[i]],d[i]))

    #Expand frequencies so we can  multiply
    for j in range(order):
        liste = list(range(f.ndim)) 
        liste.remove(ax[j])
        for i in liste:
            frequencies[j] = np.expand_dims(frequencies[j],axis=i)
            frequencies[j] = np.repeat(frequencies[j],f.shape[i],axis=i)
        fftf *=2*np.pi*1j*frequencies[j]
    return ifftn(fftf).real


# This function takes f and x, creates more datapoints s.t. f 
# can be extended to a periodic function with value 0 at end and beginning
# return newf, newx, newn, only works for one_dim function
def flattenandextendfunc_1d(f,x):
    # error handling
    if f.size!=x.size:
        print("Error: f and x do not have same size: f.size="
              +str(f.size)+",x.size="+str(x.size))
        print("Exit flattenandextendfu")
        return None
    # Get necessary data to extend f
    dx = x[1]-x[0]
    A=x[0]
    B=x[x.size-1]
    n=max(int(x.size/10),20) 
    # Create new data
    newx=np.arange(A-n*dx,B +n*dx,dx)
    newf=np.empty(newx.size, dtype=float)
    oldf=newf.copy()
    # Get slope at beg and end
    slope_start = (f[1]-f[0])/dx
    slope_end = (f[f.size-1]-f[f.size-2])/dx
    # Extend oldf
    oldf[n:n+f.size]=f
    oldf[n+f.size:]=slope_end*dx*np.arange(1, oldf[n+f.size:].size+1)+f[f.size-1]
    oldf[:n]=f[0]-slope_start*dx*np.arange(1, oldf[:n].size+1)[::-1] #[::-1] means reverse array
    # Create newf: in the middle equal to f, continuous extension such that 
    # start and endpoint are equal to one
    a=30
    half = A+(B-A)/2
    newf[newx<=half]=1.0/(1.0 + np.exp(-a*(newx[newx<=half]-(A-n/2*dx))))
    newf[newx>half]=-1.0/(1.0 + np.exp(-a*(newx[newx>half]-(B+n/2*dx))))+1.0
    newf = newf*oldf
    return newf,newx,n

def getnewsize(x):
    dx = x[1]-x[0]
    A=x[0]
    B=x[x.size-1]
    n=max(int(x.size/10),20) 
    newx=np.arange(A-n*dx,B +n*dx,dx)
    return newx.size

# Extends multidimensional function
def flattenandextendfunc_nd(f,inp):
    dim = f.ndim # get dimension
    newsize=[getnewsize(myinp) for myinp in inp]
    n=[]
    newinp=inp.copy()
    temp=[]
    liste = list(range(dim))
    # Iterate through axis: extend function for selected axis
    for axis in liste:
        liste_drop = liste.copy()
        liste_drop.remove(axis)
        
        # Calculate new shape
        shape = list(f.shape)
        shape[axis]=newsize[axis]
        shape = tuple(shape)
        temp = np.zeros(shape) #Create array with new extended shape

        start = np.zeros(dim, dtype=int) #for slicing
        end = np.zeros(dim, dtype=int)+f.shape[axis] # for slicing
        newend = np.zeros(dim, dtype=int)+newsize[axis] #for slicing new array

        # Select and save ranges in rangelist for iterating
        rangelist =[]
        for otheraxis in liste_drop:
            rangelist.append(range(f.shape[otheraxis]))  

        #Extend function for axis
        for J in itertools.product(*rangelist):
            i=0
            #Calculate slice
            for otheraxis in liste_drop:
                start[otheraxis] = J[i]
                end[otheraxis]=J[i]+1
                newend[otheraxis] = end[otheraxis]
                i+=1
            myslice =tuple(slice(*indexes) for indexes in zip(start, end))
            newslice =tuple(slice(*indexes) for indexes in zip(start, newend))

            #Extend one dimensional version
            col, newinp_axis, n_axis =flattenandextendfunc_1d(f[myslice].ravel(),newinp[axis])
            temp[newslice] = col.reshape(temp[newslice].shape)

        f = temp #Set f equal to temp    
        # Save new Input and new numbers f data
        newinp[axis]=newinp_axis
        n.append(n_axis)

    return f,newinp,n


# ### Calculate derivative with FFT and extension of a function

# inp list of arrays: e.g. inp = [x,t]
# u function to be differentiated
# axis decides which value should be differentiated
def calc_deriv_fftn_with_ext(u,inp,axis,dist):
    #Extension
    newu,newinp,n = flattenandextendfunc_nd(u,inp)
    #Define slice to get right values from extension
    start = n
    end = [sum(x) for x in zip(n, list(u.shape))] #end = n0+u.shape[0],n1+u.shape[1],...
    myslice =tuple(slice(*indexes) for indexes in zip(start, end))
    #Derivative 
    ux = calc_deriv_fftn(newu,axis,dist)
    ux = ux[myslice]
    return ux




