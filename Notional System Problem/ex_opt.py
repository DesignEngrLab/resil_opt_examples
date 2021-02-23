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
xdb = ((1e-6,100),(1e-6,100),(1e-6,100),(1e-6,100))
# xdes = [x_p, x_a, x_r, x_s]
# index:   0    1    2    3
# constants: [a,b,c,n,d,e,f]
# indss:      0,1,2,3,4,5,6
const =[1e5,1e4,1e4,10,1e6,200,500]

xres = [1,1]
#ind:   0   1
#lab:   x_b,x_c
xrb = ((1e-6,100), (1e-6,100))


x = xdes + xres
xb = xdb + xrb

def des_cost(xdes,const):
    return -const[0]*xdes[0] + const[1]*xdes[3] + const[2]/np.sqrt(max(xdes[2],1e-6))

def des_h_const(xdes):
    return xdes[0]-(xdes[3]+xdes[1])

def des_g_const(xdes):
    return np.power(10.0, -xdes[1]) - xdes[2]

def res_cost(xres,xdes,const):
    return xdes[2] * const[3] * (const[4]*xdes[1]*(xres[0]+xres[1]/2 + const[5]/max(xres[1],1e-6))+const[6]/max(xres[0],1e-6))

def x_to_totcost_mono(x, const):
    return des_cost(x, const) + res_cost(x[4:],x, const)
def x_to_totcost_alt(xdes,xres, const):
    return des_cost(xdes, const) + res_cost(xres,xdes, const)


h_con = NonlinearConstraint(des_h_const,0,0)
g_con = NonlinearConstraint(des_g_const,0,np.inf)

# just design cost opt
constraints_des = [{'type':'eq', 'fun': lambda x: x[0]-(x[3]+x[1]), 'jac': lambda x: np.array([1,-1,0,-1])}, {'type':'ineq', 'fun': lambda x: np.power(10.0, -x[1]) - x[2], 'jac': lambda x: np.array([0, -np.log(10)*np.power(10.0, -x[1]), -1, 0]) }]
result_d = minimize(des_cost, xdes, method='SLSQP', bounds = xdb, constraints =constraints_des, args = const)

# all-in-one 
#constraints = [{'type':'eq', 'fun': lambda x: x[0]-(x[3]+x[1]), 'jac': lambda x: np.array([1,-1,0,-1,0,0])}, {'type':'ineq', 'fun': lambda x: np.power(10.0, -x[1]) - x[2], 'jac': lambda x: np.array([0, -np.log(10)*np.power(10.0, -x[1]), -1, 0,0,0]) }]
#result_aao = minimize(x_to_totcost_mono, x, method='trust-constr', bounds = xb, constraints =constraints, args = const, options={'maxiter':1e3, 'verbose':True})

# bi-level
def bi_upper_level(xdes,const):
    dcost = des_cost(xdes, const)
    xres = [1,1]
    xrb =  ((0.0001,100), (0.0001,100))
    ll_result = minimize(res_cost, xres, method='trust-constr', bounds = xrb, args = (xdes, const),options={'maxiter':10})
    rcost = ll_result['fun']
    return rcost + dcost
result_bi = minimize(bi_upper_level, xdes, method='trust-constr', bounds = xdb, constraints =constraints_des, args = const, options={'maxiter':100, 'verbose':True})

# alternating
def alternating(x_init, const, constraints_des):
    xdes = x_init[0:4]
    xres = x_init[4:]
    xdb = ((0,100),(0,100),(0,100),(0,100))
    xrb = ((1e-6,100), (1e-6,100))
    for i in range(20):
        result_d = minimize(x_to_totcost_alt, xdes, method='trust-constr', bounds = xdb, constraints =constraints_des, args = (xres,const),options={'maxiter':100})
        xdes = result_d['x']
        result_r = minimize(res_cost, xres, method='trust-constr', bounds = xrb, args = (xdes, const),options={'maxiter':100})
        xres = result_r['x']
        fval = result_d['fun'] + result_r['fun']
    return xdes,xres, fval

xdes,xres, fval = alternating(x, const, constraints_des)
        


# substituting in the h-constraint: x_s = x_p + x_a / x[3] = x[0] + x[1]
#def des_cost_sub(xdes,const):
#    return -const[0]*xdes[0] + const[1]*(xdes[0] + xdes[1]) + const[2]* np.power(10.0, xdes[2])
#xdes_sub = [1,1,1]
#xdb_sub = ((0,100),(0,100),(0,100))
#
#constraints_sub = [{'type':'ineq', 'fun': lambda x: np.power(10.0, -x[1]) - x[2], 'jac': lambda x: np.array([0, -np.log(10)*np.power(10.0, -x[1]), -1]) }]
#result_sub = minimize(des_cost_sub, xdes_sub, method='trust-constr', bounds = xdb_sub, constraints =constraints_sub, args = const)
    
    