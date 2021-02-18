# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:39:43 2021

@author: dhulse
"""
import sys
sys.path.append('../')
import numpy as np
import scipy as sp
import time
import itertools
import random
from scipy.optimize import minimize

xdes = [1,1,1,1]
xdb = ((0,100),(0,100),(0,100),(0,100))
# xdes = [x_p, x_a, x_r, x_s]
# index:   0    1    2    3
# constants:
const =[1,1,1,1,1,1,1]

def des_cost(xdes,const):
    return -const[0]*xdes[0] + const[1]*xdes[3] + const[2]* np.power(10.0, xdes[2])

def des_h_const(xdes):
    return xdes[0]-(xdes[3]+xdes[1])

def des_g_const(xdes):
    return np.power(10.0, -xdes[1]) - xdes[2]

constraints = [{'type':'eq', 'fun':des_h_const}, {'type':'ineq', 'fun':des_g_const}]
result = minimize(des_cost, xdes, method='SLSQP', bounds = xdb, constraints =constraints, args = const)
    
    