import numpy as np
r_orig_ic = np.array([[0.5, 0], [2.25, 0], [0.5,1], [1.25,1], [1.5,1], [2.25,1], [0.5, 2], [2.25, 2]], dtype=float)  # shape=(nb_positions, 2)
diff = np.zeros_like(r_orig_ic)

posDisplaced = 5
dimDisplaced = 0
distanceDisplaced = 0.5
diff = np.zeros_like(r_orig_ic)
diff[posDisplaced, dimDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0,3,1,2,4,3,4,6])
j_p = np.array([1,0,4,3,5,6,7,7])
# angles
i_t = np.array([1,4,2,0,6,1,5,7,3,6])  # containing first end point
j_t = np.array([0,1,3,3,3,4,4,4,6,7])  # containing angle points2
k_t = np.array([3,0,0,6,2,5,7,1,7,4])  # containing second end point
