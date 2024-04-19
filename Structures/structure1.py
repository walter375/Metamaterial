import numpy as np

"""
    2d structure
                2 
    _/\_    0_1/ \3_5
     \/        \ /
            x   4
           0,0
"""
r_orig_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]], dtype=float)  # shape=(nb_hinges, 2)
diff = np.zeros_like(r_orig_ic)
diff[5, 0] += 0.5
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0, 1, 2, 1, 4, 3])
j_p = np.array([1, 2, 3, 4, 3, 5])
# angles
i_t = np.array([0, 2, 4, 3, 1, 4, 2, 5])  # containing first end point
j_t = np.array([1, 1, 1, 2, 4, 3, 3, 3])  # containing angle points2
k_t = np.array([2, 4, 0, 1, 3, 2, 5, 4])  # containing second end point