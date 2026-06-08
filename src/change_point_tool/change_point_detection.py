# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:19:41 2025

@author: Jiaqi
"""


import gurobipy as gp
from gurobipy import GRB
import numpy as np


def change_point_detection_fixed_num(Length, X, Delta):

    L = Length
    u_vals = np.sort(np.array(X))  # Ensure sorted data
    n = len(X)
    delta = Delta
#possible_set = 
    model = gp.Model("Segmented_CDF_Diff")

# Variables: z[i,l] ∈ {0,1}
    z = model.addVars(n, L, vtype=GRB.BINARY, name="z")
#z = model.addVars(n, L, lb=0,ub=1, vtype=GRB.CONTINUOUS, name="z")
    s = model.addVars(n, L,  vtype=GRB.CONTINUOUS, name="s")
    cdf_l = model.addVars(n, L,lb=0,ub=1, vtype=GRB.CONTINUOUS)
    t = model.addVars(n, L, lb=0, vtype=GRB.CONTINUOUS, name = 't')
    k = model.addVars(L, lb=0, ub = 1/delta, vtype=GRB.CONTINUOUS, name="y")
    b = model.addVars(n, L, n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name = 't')  
    num_l = model.addVars(n, L, vtype=GRB.CONTINUOUS)
    denom_l = model.addVars(L, vtype=GRB.CONTINUOUS)

    diff = model.addVars(n, L,lb=0,ub=1, vtype= GRB.CONTINUOUS)

    diff_dem = model.addVars(n, L, vtype= GRB.CONTINUOUS)
    y = model.addVars(n, vtype=GRB.BINARY, name="z")
# Constraints: each i must be in exactly one segment
    for i in range(n):
        model.addConstr(sum(z[i, l] for l in range(L)) == 1,name=f"segment{i}")
    
    for l in range(L):
        model.addConstr(sum(z[i, l] for i in range(n)) >= delta,name=f"segment_l{l}")

# Constraints: monotonic segment assignment (non-decreasing in l)
    for i in range(n - 1):
        for l in range(L):
            model.addConstr(z[i, l] <= sum(z[i + 1, lp] for lp in range(l, L)), name=f"monotone_{i}_{l}")

    for i in range(n):
        for j in range(i,n):
            model.addConstr(z[i,0]>= z[j,0])
            model.addConstr(z[i,1]<= z[j,1])


    indicators = [[int(X[i] <= u) for i in range(n)] for u in u_vals]
    
# Objective: sum over l=0..L-2 of CDF difference between l and l+1 at all u

    for l in range(L):
        model.addConstr(gp.quicksum(t[i, l] for i in range(n))==1)
        for u in range(n):
            model.addConstr(cdf_l[u,l] - sum(indicators[u][i] * t[i, l] for i in range(n))==0)
            model.addConstr(t[u,l]<= z[u,l])
            model.addConstr(t[u,l]<= k[l]+1/n*z[u,l]-1/n)
            model.addConstr(t[u,l]>= 1/n*z[u,l])
            model.addConstr(t[u,l]>= k[l]+z[u,l]-1)
        

    for l in range(L):
        for u in range(n):
            for i  in range(n):
                model.addConstr(b[i,l,u]<=diff[u,l], name=f"t_cdf_{i}_{l}_{u}")
                model.addConstr(b[i,l,u]<=z[i,l], name=f"t_z_{i}_{l}_{u}")
                model.addConstr(b[i,l,u]>=z[i,l]+diff[u,l]-1, name=f"t_1_{i}_{l}_{u}")


    for l in range(L):
        for u in range(n):
            model.addConstr(diff[u,l]+cdf_l[u,l] ==1)
            model.addConstr(s[u,l] == sum(indicators[u][i]*b[i,l,u] for i in range(n)))
        
        
        
    objective = sum(s[u,l] for u in range(n) for l in range(L))
    model.setObjective(objective, GRB.MINIMIZE)

    
    model.Params.OutputFlag = 1

    model.optimize()

   
    z_sol = np.zeros((n, L))
    s_sol = np.zeros((n, L))
    cdf_l_sol = np.zeros((n, L))
    cdf_lp1_sol = np.zeros((n, L))
    num_l_sol = np.zeros((n, L))
    denom_l_sol = np.zeros((L))
    if model.status == GRB.OPTIMAL:
        for i in range(n):
            for l in range(L):
                z_sol[i, l] = z[i, l].X
                s_sol[i,l]= s[i,l].X
                cdf_l_sol[i,l]= cdf_l[i,l].X
                num_l_sol[i,l]= num_l[i,l].X
                denom_l_sol[l]= denom_l[l].X
            
    

    #print("Segment assignments (z):\n", model.ObjVal)
    col_sums = np.sum(z_sol, axis=0)

    Z_1 = np.cumsum(col_sums)

    return Z_1