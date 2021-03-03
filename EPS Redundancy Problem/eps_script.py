# -*- coding: utf-8 -*-
"""
Optimization of EPS System
"""

import epsOpt as eo
import ibfm
import importlib

importlib.reload(ibfm)
importlib.reload(eo)

e1=ibfm.Experiment('eps_simple')
e1.run(1)

functions, fxnscores, fxnprobs, failutility, fxncost = eo.scorefxns(e1)

factor=1 # factor adjusts the amount of penalty on the redundancies

fxnreds, newfailutility=eo.optRedundancy(functions, fxnscores, fxnprobs, fxncost,factor)

print(fxnreds)

