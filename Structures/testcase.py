"""
    0---1---2
"""
import numpy as np
# test case
r_orig_ic = np.array([[0,0],[1,0],[2,0]], dtype=float)
i_p = np.array([0,1])
j_p = np.array([1,2])

posDisplaced = 2
dimDisplaced = 0
distanceDisplaced = 1
diff = np.zeros_like(r_orig_ic)
diff[posDisplaced, dimDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff