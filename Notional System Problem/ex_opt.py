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
from scipy.optimize import NonlinearConstraint

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


h_con = NonlinearConstraint(des_h_const,0,0)
g_con = NonlinearConstraint(des_g_const,0,np.inf)

constraints = [{'type':'eq', 'fun': lambda x: x[0]-(x[3]+x[1]), 'jac': lambda x: np.array([1,-1,0,-1])}, {'type':'ineq', 'fun': lambda x: np.power(10.0, -x[1]) - x[2], 'jac': lambda x: np.array([0, -np.log(10)*np.power(10.0, -x[1]), -1, 0]) }]
result = minimize(des_cost, xdes, method='SLSQP', bounds = xdb, constraints =constraints, args = const)


# substituting in the h-constraint: x_s = x_p + x_a / x[3] = x[0] + x[1]
#def des_cost_sub(xdes,const):
#    return -const[0]*xdes[0] + const[1]*(xdes[0] + xdes[1]) + const[2]* np.power(10.0, xdes[2])
#xdes_sub = [1,1,1]
#xdb_sub = ((0,100),(0,100),(0,100))
#
#constraints_sub = [{'type':'ineq', 'fun': lambda x: np.power(10.0, -x[1]) - x[2], 'jac': lambda x: np.array([0, -np.log(10)*np.power(10.0, -x[1]), -1]) }]
#result_sub = minimize(des_cost_sub, xdes_sub, method='trust-constr', bounds = xdb_sub, constraints =constraints_sub, args = const)
    
    