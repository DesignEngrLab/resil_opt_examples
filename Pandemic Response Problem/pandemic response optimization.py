# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 00:10:33 2020

@author: zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:02:21 2020

@author: zhang
"""
import pandas as pd
import numpy as np
import time
import csv
from scipy.optimize import minimize
import matplotlib
from matplotlib import pyplot as plt 
from scipy.optimize import differential_evolution


a=0.2
b=10

t=160
#step=int(t*1/h)

def DiseaseModel(x0):
    
    
    S=list(range(0,t)); S[0]=900
    I=list(range(0,t)); I[0]=100
    R=list(range(0,t)); R[0]=0
    N=S[0]+I[0]+R[0]
        
    PL1=0
    PL2=0
    nom=0
    
    
    a2=x0[0]
    n=x0[1]
    nms = n
    v=x0[2]
    m=x0[3]
    alpha=x0[4]
    IR=x0[5]
    
    c=m/m
#    print(a2,n,v,m,alpha,IR)
    b2=b
    vc=0
       
    infect_rate=list(range(0,t))
    recover_rate=list(range(0,t))
    
    infect_rate[0]= a * S[0] * I[0] / N
    recover_rate[0]= I[0]/b
    
    state=list(range(0,t))
    state[0]='nom'
    for i in range(1,t):
    # PL1 and PL2 both triggered
#        print('\n')
         
        if I[i-1]/N > alpha and infect_rate[i-1] > IR:
            c =( m+nms )/ m
            nms = m+n
            
            infect_rate[i]= a2 * (S[i-1]) * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b2
            
            S[i]= S[i-1] -  (infect_rate[i] ) - v
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i])
            R[i]= R[i-1] +  (recover_rate[i]) + v
                   
            PL1=PL1+1
            PL2=PL2+1
            state[i]='PL1&PL2'
            vc=vc+v
            
    # PL1 triggered
        elif I[i-1]/N > alpha :
            infect_rate[i] = a2 * (S[i-1]) * I[i-1] / N
            recover_rate[i] = I[i-1]/b2
            
            S[i]= S[i-1] -  (infect_rate[i] ) - v
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i] )
            R[i]= R[i-1] +  (recover_rate[i]) + v
                        
            PL1=PL1+1
            state[i]='PL1'
            vc=vc+v
    # PL2 triggered     
        elif  infect_rate[i-1] > IR : 
            c =( m+nms )/ m
            nms = m+n
            
            infect_rate[i]= a * (S[i-1]) * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b2
        
            S[i]= S[i-1] -  (infect_rate[i])
            I[i]= I[i-1] +  (infect_rate[i] - recover_rate[i])
            R[i]= R[i-1] +  (recover_rate[i])
            
            PL2=PL2+1
            state[i]='PL2'
    #     nominal state
        else:
            infect_rate[i]= a * S[i-1] * I[i-1] / N
            recover_rate[i]= c * I[i-1]/b
            
            S[i]= S[i-1] -  (infect_rate[i])
            I[i]= I[i-1] +  (infect_rate[i] - c * I[i-1]/b)
            R[i]= R[i-1] +  (c * I[i-1]/b)
            
            nom=nom+1  
            state[i]='nom'
    # treatment fee for each people            
    H=100000
    # average expense for each people
    E=10000
    # PL1 lasting time
    T=PL1 
    # salary for each medical people per day
    Em=10000
    # extra medical people total working time
    Tm=n * (t-1+t-PL2)*PL2/2        
    
    # cost of infections + cost of reduce spread measure + cost of treatment measure
    totalcost=(R[-1]-vc)*H + (a-x0[0])* N * E * T +  Em * Tm
    return totalcost, S , I , R, PL1, PL2, nom,infect_rate,recover_rate, state,vc

def objective(x0):
    totalcost,_,_,_,_,_,_,_,_,_,_ = DiseaseModel(x0)
    return totalcost
def objective2(x0, args):
    totalcost,_,_,_,_,_,_,_,_,_,_ = DiseaseModel(x0)
    args['timehist'] = args['timehist'] +[time.time()-args['starttime']]
    args['fhist'] = args['fhist'] + [totalcost]
    return totalcost


starttime = time.time()
timehist = []
fhist = []

def track_opthist(xk, convergence):
    print(objective(xk))
    

# # 'a': x0[0] ,'n':x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5]
#x0 = np.array([0.1 , 10 , 5 , 10 , 0.15 , 1 ])


## nominal state
#x0 = [0.2 , 0 , 0 , 10 , 0 , 0 ]   
## under PL1 only
#x0 = [0.1 , 2 , 5 , 10 , 0.05 , 100 ]    
## under PL2 only
#x0 = [0.1 , 2 , 5 , 10 , 100 , 2 ]
# under PL1&PL2 
#x0 = [0.1 , 2 , 5 , 10 , 0.05 , 5 ]
#x0= [0.07 , 1 , 9.6 , 9 , 0.329 , 4.9]
    
## Starting point?
#x0 = [1,0.5,1e-4, 0.5,1,1]
x0 = [0.1 , 2 , 9 , 9, 1000 , 1000 ] 
result_x0=list(DiseaseModel(x0))
#print(result0[3])
x=list(range(0,t))


print(result_x0[1][-1]+result_x0[2][-1]+result_x0[3][-1])
print('PL1:',result_x0[4],'PL2:',result_x0[5],'nom:',result_x0[6])

print('totalcost',result_x0[0])
print('total number of people get vaccine:' ,result_x0[10])


#totalcost, S , I , R, PL1, PL2, nom,infect_rate,recover_rate, state,vc
dataframe = pd.DataFrame({'t': range(0,160)  ,'S':result_x0[1],'I':result_x0[2],'R':result_x0[3],'infect_rate':result_x0[7],'recover_rate':result_x0[8],'state':result_x0[9]})
dataframe.to_csv("test_new_PL1.csv",index=False,sep=',')


# need a better objective - currently the bounds are the only thing keeping this from
# not having a policy response at all
bounds = [(0, 0.2), (1, 5),(8, 10),(8, 10),(0, 200),(0, 500)]
starttime= time.time()
result = differential_evolution(objective, bounds, maxiter=10000, callback = track_opthist)
endtime = time.time() - starttime
print("-----------------")

##
##print(result.x, result.fun)
# # 'a': x0[0] ,'n':x0[1] ,'v' : x0[2] ,'m': x0[3], 'alpha': x0[4] , 'IR':x0[5]
# 
print('a2=',result.x[0])
print('n=',result.x[1])
print('v=',result.x[2])
print('m=',result.x[3])
print('alpha=',result.x[4])
print('IR=',result.x[5])
print('cost=',result.fun)
print(result.nit)

result_opt = list(DiseaseModel(result.x))

fig = plt.figure()
plt.plot(x,result_x0[1],'k', linestyle=":", label='Susceptible')
plt.plot(x,result_x0[2],'r', linestyle=":", label='Infected')
plt.plot(x,result_x0[3],'g', linestyle=":", label='Recovered')

plt.plot(x,result_opt[1],'k', label='Susceptible ($x^*$)')
plt.plot(x,result_opt[2],'r', label='Infected ($x^*$)')
plt.plot(x,result_opt[3],'g', label='Recovered ($x^*$)')
plt.legend()
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.grid()
plt.show()
fig.savefig('pandemic_behavior.pdf', format="pdf", bbox_inches = 'tight', pad_inches = 0.0)


stop=0
for i in range(0,t):
    if result_opt[2][i] < 5:
        stop=i; break
    
print('stop day:',stop)
print('total infected people:',result_opt[3][-1]-result_opt[10])


fig = plt.figure(figsize=(3, 2.5))
# have to grab this from the output of the algorithm and place in opt_results.csv

opthist = pd.read_csv("opt_results.csv")
thist = np.linspace(0, endtime, len(opthist))

plt.plot(thist, opthist, linewidth=3)
plt.xlabel("Computational Time (s)")
plt.ylabel("Best Solution")
plt.grid()
fig.savefig('pandemic_opt.pdf', format="pdf", bbox_inches = 'tight', pad_inches = 0.0)

#print(result0[9])