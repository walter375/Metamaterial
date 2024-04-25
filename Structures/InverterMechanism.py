import numpy as np

r_orig_ic = np.array([[0, 0], [1, 0], [0.75, 0.5], [0.01,1], [2,1], [2.25,1], [0.75, 1.5], [0, 2], [1, 2]], dtype=float)  # shape=(nb_hinges, 2)
diff = np.zeros_like(r_orig_ic)
diff[3,0] += 0.3
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0, 0, 1, 1, 2, 3, 4, 6, 6, 7, 8])
j_p = np.array([1, 2, 2, 4, 3, 6, 5, 7, 8, 8, 4])
# angles
i_t = np.array([2,0,1,0,3,2,1,4,8,6,8,6,7,3,8,5])  # containing first end point
j_t = np.array([0,1,2,2,2,1,4,8,6,3,7,8,6,6,4,4])  # containing angle points2
k_t = np.array([1,2,0,3,1,4,8,6,3,2,6,7,8,7,5,1])  # containing second end point



