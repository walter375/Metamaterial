import numpy as np

"""
2d structure
               1----2----3
                \  / \  /
             0---45   67---8
                /  \ /  \
               9---10---11
"""
# positions
r_orig_ic = np.array(
    [[0, 0.5], [1, 1], [2.5, 1], [4, 1], [1.65, 0.5], [1.85, 0.5], [3.15, 0.5], [3.35, 0.5], [4.5, 0.5], [1, 0], [2.5, 0],
     [4, 0]], dtype=float)  # shape=(nb_positions, 2)
posDisplaced = np.array((0,8))
dimDisplaced = 0
distanceDisplaced =1.0
diff = np.zeros_like(r_orig_ic)
diff[0, dimDisplaced] -= distanceDisplaced
diff[8, dimDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([1, 2, 0, 1, 2, 2, 3, 7, 4, 5, 6, 7, 9, 10])
j_p = np.array([2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 10, 11, 10, 11])
# angles
i_t = np.array([5, 1, 6, 7, 2,1,0,2,2,8,11,10,5,6,11,7])  # containing first end point
j_t = np.array([1, 2, 2, 2, 3,4,4,8,6,7,7,9,10,10,10,11])  # containing angle points2
k_t = np.array([2, 6, 7, 3, 8,0,9,10,10,3,8,4,9,5,6,10])  # containing second end point
