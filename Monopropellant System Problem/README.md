# Monopropellant Optimization Model

This folder demonstrates the optimization of a resilience policy for a monopropellant system using an evolutionary algorithm. This system is described in: 

Hulse, D., Hoyle, C., Goebel, K., & Tumer, I. Y. (2019). Quantifying the resilience-informed scenario cost sum: A value-driven design approach for functional hazard assessment. Journal of Mechanical Design, 141(2).

Running this code requires the following dependencies:
- matplotlib
- numpy
- networkx

Additionally, it requires the latest copy of ibfm.py in the folder, which can be downloaded from "https://github.com/DesignEngrLab/IBFM"

The files in this repository are:
- monoprop_script.py, which demonstrates the use of the evolutionary algorithm
- monopropOpt.py, which contains the evolutionary algorithm, methods for changing the model, and the objective functions
- monoprop.ibfm, which is the model of the system. Other .ibfm files (e.g. modes.ibfm, conditions.ibfm) define the components of this model
