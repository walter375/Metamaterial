"""
    0---1---2---3
"""
import numpy as np
# test case
#r_orig_ic = np.array([[0,0],[0,1],[0,2],[0,3]], dtype=float)
#r_orig_ic = np.array([[0,0],[1,0],[2,0],[3,0]], dtype=float)
r_orig_ic = np.array([[0,0],[1,1],[2,2],[3,3]], dtype=float)
i_p = np.array([0,1,2])
j_p = np.array([1,2,3])

posDisplaced = 3
dimDisplaced = 1
distanceDisplaced = 0.5
diff = np.zeros_like(r_orig_ic)
if dimDisplaced == 0:
    diff[posDisplaced, 0] += distanceDisplaced
if dimDisplaced == 1:
    diff[posDisplaced, 1] += distanceDisplaced
if dimDisplaced == 2:
    diff[posDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff