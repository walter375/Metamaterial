import numpy as np
# hinges at points 3, 6, 7, 8
r_orig_ic = np.array([[0,0], [2,0], [2.75,0], [1,1], [2.6,1], [3,1], [0,2], [2,2], [1,3], [2.6,3], [3,3], [0,4], [2,4], [2.75,4]], dtype=float)  # shape=(nb_positions, 2)

posDisplaced = 6
dimDisplaced = 0
distanceDisplaced = 0.8
diff = np.zeros_like(r_orig_ic)
diff[posDisplaced, dimDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0,3,1,1,4,2,4,6,3,7,6,8,7,11,8,12,9,9,13,12])
j_p = np.array([3,1,2,4,2,5,5,3,7,4,8,7,9,8,12,9,10,13,10,13])
# angles
i_t = np.array([6,0,1,4,7,3,2,4,1,5,2,2,5,4,7,8,7,8,6,11,12,7,9,12,8,9,13,12,10,13,9])  # containing first end point
j_t = np.array([3,3,3,1,4,7,1,2,4,2,5,4,4,7,9,6,3,7,8,8,8,8,7,9,12,12,9,13,9,10,13])  # containing angle points2
k_t = np.array([0,1,7,3,1,4,4,1,2,4,4,5,7,9,10,3,6,3,7,6,11,12,8,7,9,13,12,9,13,9,10])  # containing second end point
