# EPS Optimization Model

This folder demonstrates the optimization of redundancy of an EPS system using a decomposition strategy. This system is described in: 

Hulse, D., Hoyle, C., Tumer, I. Y., & Goebel, K. (2021). How Uncertain Is Too Uncertain? Validity Tests for Early Resilient and Risk-Based Design Processes. Journal of Mechanical Design, 143(1).

Running this code requires the following dependencies:
- numpy
- networkx

Additionally, it requires the latest copy of ibfm.py in the folder, which can be downloaded from "https://github.com/DesignEngrLab/IBFM"

The files in this repository are:
- esp_script.py, which demonstrates the optimization
- epsOpt.py, which contains the  algorithm and objective functions
- eps.ibfm, which is the model of the system. Other .ibfm files (e.g. modes.ibfm, conditions.ibfm) define the components of this model
