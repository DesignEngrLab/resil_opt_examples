# resil_opt_examples

[![DOI](https://zenodo.org/badge/339532572.svg)](https://zenodo.org/badge/latestdoi/339532572)

This repository contains examples of resilience optimization problems for the study of optimization approaches and strategies.

The models are:

- `/Notional System Problem`, which provides a basic demonstration of an Integrated Resilience Optimization problem where all variables are continuous.

- `/Cooling Tank Problem`, which provides a demonstration of an Integrated Resilience Optimization problem with mixed variable types where the model is defined in a (fmdtools) dynamical resilience simulation. 

- `/Drone Problem`, which provides a demonstration of an Integrated Resilience Optimization problem with a decomposable resilience model defined in a (fmdtools) dynamical resilience simulation. 

- `/EPS Redundancy Problem`, which demonstrates a Resilience-based Design Optimization formulation where the optimization is decomposed to each redundancy. It uses an IBFM simulation.

- `/Monopropellant System Problem`, which demonstrates a Resilience Policy Optimization formulation solved with an evolutionary algorithm. It uses an IBFM simulation.

- `/Pandemic Response Problem`, which demonstrates a Resilience Policy Optimization formulation solved with a differential evolution algorithm. It is defined in a standalone model.

### Contributors

Daniel Hulse 

Hongyang Zhang - pandemic model

Arpan Biswas - drone model
