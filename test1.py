# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:02:38 2023

@author: Hansol
"""

import cvxpy as cp
import numpy as np

#I tried to simulate my inputs for you. 
#Here the problem manifests itself already at m, n = 1600, 4950.
np.random.seed(0)
m, n= 1600, 4950
A = np.round(np.random.rand(m, n)/(1-1/m**(1/2))/2)*255*(m**(1/2)/1800)**1.1
b = 255*(np.random.randn(m))

x = cp.Variable(n, integer=True)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = []
constraints.extend([x >= 0.0, x <= 1])
prob = cp.Problem(objective,constraints)
prob.solve(solver=cp.SCIP,verbose=True,scip_params={"limits/time": 50})
print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("Non-zero ",sum([i for i in x.value]))