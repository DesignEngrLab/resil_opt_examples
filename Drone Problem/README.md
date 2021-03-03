# Drone Optimization Problem

This folder demonstrates the optimization of the architecture, flight profile, and resilience policy of a multirotor drone. 

The simulation for this model was developed in fmdtools. To install this toolkit and the required dependencies, use 'pip install fmdtools' or see https://github.com/DesignEngrLab/fmdtools.
This model additionally requires:
- scipy
- shapely
- seaborn
- mpl_toolkits.mplot3d

The files in this folder are:
- Case Study Explanation.ipynb, which is a jupyter notebook that demonstrates the simulation and optimization of the drone model
- drone_opt.py, which contains the algorithms and objective functions for the model
- drone_model.py, which is the model file for the drone system.
