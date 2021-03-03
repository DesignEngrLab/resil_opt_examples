# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:34:44 2017

@author: hulsed
"""
import numpy as np
import ibfm
import networkx as nx

#Creates function variants in corresponding files
def createVariants():
    
    file=open('ctlfxnvariants.ibfm', mode='w')
    variants=[3,3,3]
    options=list(range(1,variants[0]+1))
    
    for i in range(variants[0]):
        for j in range(variants[1]):
            for k in range(variants[2]):
                file.write('function ControlSig'+str(i+1)+str(j+1)+str(k+1)+'\n')
                file.write('\t'+'mode 1 Operational EqualControl \n')
                file.write('\t'+'mode 2 Operational IncreaseControl \n')
                file.write('\t'+'mode 3 Operational DecreaseControl \n')
                
                toremove=i+1
                inmodes=list(filter(lambda x: x!=toremove,options))
                inmodestr=' '.join(str(x) for x in inmodes)
                
                file.write('\t'+'condition '+inmodestr+' to '+str(toremove)+' LowSignal'+'\n')
                
                toremove=j+1
                inmodes=list(filter(lambda x: x!=toremove,options))
                inmodestr=' '.join(str(x) for x in inmodes)
                
                file.write('\t'+'condition '+inmodestr+' to '+str(toremove)+' HighSignal'+'\n')
                
                toremove=k+1
                inmodes=list(filter(lambda x: x!=toremove,options))
                inmodestr=' '.join(str(x) for x in inmodes)
                
                file.write('\t'+'condition '+inmodestr+' to '+str(toremove)+' NominalSignal'+'\n')
                
    file.close()
    return 0

def randFullPolicy(controllers, conditions):
    FullPolicy=np.random.randint(3,size=(controllers,conditions))+1
    return FullPolicy

def randController(FullPolicy):
    controllers,conditions=FullPolicy.shape
    randcontroller=np.random.randint(controllers)
    FullPolicy[randcontroller]=np.random.randint(3,size=conditions)+1    
    return FullPolicy

def randCondition(FullPolicy):
    controllers,conditions=FullPolicy.shape
    randcontroller=np.random.randint(controllers)
    randcondition=np.random.randint(conditions)
    FullPolicy[randcontroller][randcondition]=np.random.randint(3)+1
    return FullPolicy

def initPopulation(pop,controllers, conditions):
    Population=np.random.randint(3,size=(pop,controllers,conditions))+1
    return Population
    
def permutePopulation(Population):
    pop,controllers,conditions=Population.shape
    frac=0.8
    
    Population2=Population.copy()
    
    for i in range(pop):
        oldPolicy=Population[i]
        
        rand=np.random.rand()
        if rand>frac:
            newPolicy=randController(oldPolicy)
        else:
            newPolicy=randCondition(oldPolicy)
            
        Population2[i]=newPolicy
    return Population2

def evalPopulation(Population, experiment):
    pop,controllers,conditions=Population.shape
    fitness=np.ones(pop)
    
    for i in range(pop):
        actions, instates, utilityscores, designcost=evaluate(Population[i],experiment)
        fitness[i]=sum(utilityscores)+designcost
    
    return fitness

def selectPopulation(Population1, fitness1, Population2, fitness2, pop):
    
    totfitness=np.append(fitness1,fitness2)
    totpopulation=np.append(Population1.copy(), Population2.copy(), axis=0)
    
    popkey=totfitness.argsort()
    sortpopulation=totpopulation[popkey]
    sortfitness=totfitness[popkey]
    
    newpopulation=sortpopulation[pop:int(2*pop)]
    newfitness=sortfitness[pop:int(2*pop)]
    return newpopulation, newfitness
      
def EA(pop,generations, controllers, conditions, experiment):
    Population1=initPopulation(pop,controllers, conditions)
    Population1[0]=np.array([[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    
    fitness1=evalPopulation(Population1, experiment)
    
    Population2=permutePopulation(Population1.copy())
    fitness2=evalPopulation(Population2, experiment)
    
    fithist=np.ones(generations)
    
    for i in range(generations):
        Population2=permutePopulation(Population1.copy())
        fitness2=evalPopulation(Population2, experiment)
        
        Population1,fitness1=selectPopulation(Population1.copy(), fitness1, Population2.copy(), fitness2, pop)
        
        maxfitness=max(fitness1)
        bestsolloc=np.argmax(fitness1)
        bestsol=Population1[bestsolloc]
        fithist[i]=maxfitness        
    return maxfitness, bestsol, fithist
    
#Initializes the Policy    
def initFullPolicy(controllers, conditions):
    FullPolicy=np.ones([controllers,conditions], int)
    return FullPolicy

def evaluate2(experiment):
    scenarios=len(experiment.results)
    actions=[]
    instates=[]
    scores=[]
    probs=[]
    fxncosts=[]
    
    nominalstate=experiment.model.nominal_state
    nominalscore=scoreNomstate(nominalstate)
    
    fxns=experiment.model.graph.nodes()
    
    for fxn in fxns:
        
        try:
            fxncosts+=[float(fxn._cost[0])]
        except ValueError:
            fxncosts+=[0.0]
    
    designcost=-sum(fxncosts)
    
    for scenario in range(scenarios):
        scores+=[scoreScenario(experiment,scenario, nominalstate)]
        
        prob1=list(experiment.scenarios[scenario].values())[0].prob
        nloc=prob1.find('n')
        prob= float(prob1[:nloc]+'-'+prob1[nloc+1:])
        probs=probs+[prob]
        
    probabilities=np.array(probs)

    #probability of the nominal state is prod(1-p_e), for e independent events
    nominalprob=np.prod(1-probabilities)

    actions+=[trackNomActions(nominalstate)]
    instates+=[trackNomFlows(nominalstate)]
    scores+=[nominalscore]
    probabilities=np.append(probabilities, nominalprob)
    
    
    failcost=sum(scores*probabilities)
    
    utility = failcost + designcost
    
    return  failcost, designcost, utility, scores

#Takes the policy, changes the model, runs it, tracks the actions, and gives 
#a utility score for each scenario
def evaluate(FullPolicy,experiment):
    graph=reviseModel(FullPolicy, experiment)
    #importlib.reload(ibfm)
    newexp= ibfm.Experiment(graph)
    newexp.run(1)
    
    scenarios=len(newexp.results)
    actions=[]
    instates=[]
    scores=[]
    probs=[]
    
    nominalstate=newexp.model.nominal_state
    nominalscore=scoreNomstate(nominalstate)
    
    for scenario in range(scenarios):
        actions+=[trackActions(newexp,scenario)]
        instates+=[trackFlows(newexp,scenario)]
        scores+=[scoreScenario(newexp,scenario, nominalstate)]
        
        prob1=list(newexp.scenarios[scenario].values())[0].prob
        nloc=prob1.find('n')
        prob= float(prob1[:nloc]+'-'+prob1[nloc+1:])
        probs=probs+[prob]
        
    probabilities=np.array(probs)

    #probability of the nominal state is prod(1-p_e), for e independent events
    nominalprob=np.prod(1-probabilities)

    actions+=[trackNomActions(nominalstate)]
    instates+=[trackNomFlows(nominalstate)]
    scores+=[nominalscore]
    probabilities=np.append(probabilities, nominalprob)
    
    designcost=PolicyCost(FullPolicy)
    
    utilityscores=scores*probabilities
    
    return actions, instates, utilityscores, designcost

#Revises the model based on the policy, creating a new graph to be used in ibfm        
def reviseModel(FullPolicy, exp):
    graph=exp.model.graph.copy()
    nodes=graph.nodes()
    edges=graph.edges()
    functions=['controlG2rate','controlG3press','controlP1effort','controlP1rate']
    newgraph=nx.DiGraph()
    
    for node in nodes:
        name=str(node)
        fxn=graph.nodes[node]['function']
        newgraph.add_node(name, function=fxn)
        if name in functions:
            loc=functions.index(name)
            policy=FullPolicy[loc]    
            ctlfxn='ControlSig'+str(policy[0])+str(policy[1])+str(policy[2])
            newgraph.nodes[name].update({'function': ctlfxn})
              
    for edge in edges:
        prev=str(edge[0])
        new=str(edge[1])
        flowobj=graph.get_edge_data(prev,new)[0]['attr_dict']
        flowtype=list(flowobj.values())[0]
        newgraph.add_edge(prev,new,flow=flowtype)
        #newgraph.edge[prev]={new: {'flow': flowtype}}
    return newgraph

#Track the actions taken for the nominal state
def trackNomActions(nominal_state):
    #functions of concern--the controlling functions
    functions=['controlG2rate','controlG3press','controlP1effort','controlP1rate']
    mode2actions={'EqualControl': 0, 'IncreaseControl': 1, 'DecreaseControl': 2}
    
    actions=[]
    #find actions taken
    for function in functions:
        mode=str(nominal_state[function])
        actions+=[mode2actions[mode]] 
    return actions

#Track the actions taken (given which scenario it is in a list) for the experiment
def trackActions(exp, scenario):
    #functions of concern--the controlling functions
    functions=['controlG2rate','controlG3press','controlP1effort','controlP1rate']
    mode2actions={'EqualControl': 0, 'IncreaseControl': 1, 'DecreaseControl': 2}
    numstates=len(exp.getResults())
    
    actions=[]
    #find actions taken
    for function in functions:
        mode=str(exp.results[scenario][function])
        actions+=[mode2actions[mode]] 
    return actions

#Track flows going into the functions for the nominal state, as per trackFlows()
def trackNomFlows(nominal_state):
    #flows of concern--inputs to the controllers    
    condition2state={'Negative':0,'Zero': 0,'Low': 0,'High': 1,'Highest': 1,'Nominal': 2}
    
    flowtypesraw=list(nominal_state.keys())
    flowtypes=[str(j) for j in flowtypesraw]
    flowstates=list(nominal_state.values())
    
    flownum=len(flowtypes)
    instates=[]
    
    #find states entered
    for i in range(flownum):
        if flowtypes[i]=='Signal':
            if i%2==0:
                incondition=str(flowstates[i][0])
                instate=condition2state[incondition]
                instates+=[instate]
                
    return instates

#Track flows going into the functions for each scenario 
#(given which scenario it is in a list) for the experiment
#This is used for seeing how an action in one controller changes the state
#for the next controller. However, it may only work if the controllers are 
#oriented in a chain
#NOTE: May ONLY work if only signals are to controllers
def trackFlows(exp, scenario):
    #flows of concern--inputs to the controllers    
    condition2state={'Negative':0,'Zero': 0,'Low': 0,'High': 1,'Highest': 1,'Nominal': 2}
    
    flowtypesraw=list(exp.results[scenario].keys())
    flowtypes=[str(j) for j in flowtypesraw]
    flowstates=list(exp.results[scenario].values())
    
    flownum=len(flowtypes)
    instates=[]
    
    #find states entered
    for i in range(flownum):
        if flowtypes[i]=='Signal':
            if i%2==0:
                incondition=str(flowstates[i][0])
                instate=condition2state[incondition]
                instates+=[instate]
                
    return instates

#creates a cost for the design provided it enables certain parts of the policy
def PolicyCost(FullPolicy):
    increasecost=[-50000, -50000, -50000, -50000]
    decreasecost=[-5000, -50000, -50000, -5000]
    increasecapcost=[-5000000, 0, 0, -20000000]
    
    controllers=len(FullPolicy)
    
    cost=0.0
    
    for controller in range(controllers):
        if any(FullPolicy[controller]==[2,2,2]):
            cost+=increasecost[controller]
            cost+=increasecapcost[controller]
        if any(FullPolicy[controller]==[3,3,3]):
            cost+=decreasecost[controller]
        
    return cost

#Score a scenario (given which scenario it is in a list)
def scoreScenario(exp, scenario, Nominalstate):
    time2coeff={'beginning':0.0,'early':0.1,'midway':0.3,'late':0.6,'end':0.9}
    
    nomscore=scoreNomstate(Nominalstate)
    endscore=scoreEndstate(exp, scenario)
    
    
    
    time=list(exp.scenarios[scenario].values())[0].when
    cnom=time2coeff[time]
    cend=1.0-cnom
    
    scenscore=cnom*nomscore+cend*endscore
    
    return scenscore

#Score the nominal state
def scoreNomstate(Nominalstate):
    functions=['exportT1']
    Flow="Thrust"
    
    flowsraw=list(Nominalstate.keys())
    flows=[str(j) for j in flowsraw]
    states=list(Nominalstate.values())
    loc=flows.index(Flow)
    
    effort=int(states[loc][0])
    rate=int(states[loc][1])
    statescore=scoreFlowstate(rate,effort)
    
    return statescore

#Score an endstate for the experiment
def scoreEndstate(exp, scenario):
    Flow="Thrust"
    
    flowsraw=list(exp.results[scenario].keys())
    flows=[str(j) for j in flowsraw]
    states=list(exp.results[scenario].values())
    loc=flows.index(Flow)
        
    effort=int(states[loc][0])
    rate=int(states[loc][1])
    
    statescore=scoreFlowstate(rate,effort)
    
    return statescore

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
    score=50e6*qualfunc[effort][rate]
    return score

