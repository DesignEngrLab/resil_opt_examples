# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:34:44 2017

@author: hulsed
"""
import numpy as np


def scoreEndstates(exp, scenario):
    functions=['exportT1']
    Flows=["OpticalEnergy","MechanicalEnergy","DesiredHeat"]
    statescores=[]
    
    for Flow in Flows:
    
        flowsraw=list(exp.results[scenario].keys())
        flows=[str(j) for j in flowsraw]
        states=list(exp.results[scenario].values())
        loc=flows.index(Flow)
        
        effort=int(states[loc][0])
        rate=int(states[loc][1])
    
        statescores=statescores+[scoreFlowstate(rate,effort)]
    
    statescore=sum(statescores)
    
    return statescore

#Score function for a given flow state.
def scoreFlowstate(rate, effort):
    qualfunc = [[-90.,-80.,-70.,-85.,-100.],
            [-80., -50., -20, -15, -100.],
            [-70., -20.,  0., -20., -100.],
            [-85., -10, -20., -50.,-110.],
            [-100., -100., -100.,-110.,-110.]]
    #1e6
    #10
    score=100*qualfunc[effort][rate]
    return score


#Individually scores functions based on their failure impact
def scorefxns(exp):
    
    exp.run(1)
    scenarios=len(exp.results)
    
    functions=[]
    scores=[]
    probs=[]
    
    #initialize dictionary
    functions=exp.model.graph.nodes()
    fxnscores={}
    fxnprobs={}
    failutility={}
    fxncost={}
    
    for fxn in functions:
        fxnscores[fxn]=[]
        fxnprobs[fxn]=[]
        failutility[fxn]=0.0
        try:
            fxncost[fxn]=float(fxn._cost[0])
        except ValueError:
            fxncost[fxn]=0.0
    #map each scenario to its originating function
    for scenario in range(scenarios):
        function=list(exp.scenarios[scenario].keys())[0]
        
        
        prob1=list(exp.scenarios[scenario].values())[0].prob
        nloc=prob1.find('n')
        prob= float(prob1[:nloc]+'-'+prob1[nloc+1:])
        
        probs=probs+[prob]
        probs+=[prob]

        score=scoreEndstates(exp,scenario)
        scores+=[score]
        
        fxnscores[function]+=[score]
        fxnprobs[function]+=[prob]
        failutility[function]=failutility[function] + score*prob
    #map each function to the utility of making it redundant
    
    return functions, fxnscores, fxnprobs, failutility, fxncost

def optRedundancy(functions, fxnscores, fxnprobs, fxncost, factor):  
    fxnreds={}
    newfailutility={}
    
    for function in functions:
        fxnreds[function]=0
        newfailutility[function]=0.0
    
    ufunc=0.0
    
    for function in functions:
        probs=np.array(fxnprobs[function])
        scores=factor*np.array(fxnscores[function])
        cost=fxncost[function]
        
        ufunc=sum(scores*probs)-cost
        converged=0
        n=0
        #print(ufunc)
        
        if cost == 0:
            n=1
        else:
            while not(converged):
                n+=1
                newufunc=sum(scores*probs**(n+1))-cost*(n-1)
                #print(newufunc)
                
                if newufunc >= ufunc:
                    ufunc=newufunc
                else:
                    converged=1
                    
                if n>150:
                    break
        
        fxnreds[function]=n
        newfailutility[function]=ufunc
    return fxnreds, newfailutility
                
    
