import numpy as np
# hinges at points 3, 6, 7, 8
r_orig_ic = np.array([[0,0], [0,1.75], [0,2.25], [0,4], [1.5,1], [1.5,3], [2,0.5], [2,3.5], [2.25,1.85], [2.25,2.15], [2.5,1.5], [2.5,2.5], [2.75,1.5], [2.75,2.5], [3,0.5], [3,3.5], [3.5,1], [3.5,3], [3.75,1.5], [3.75,2.5]], dtype=float)  # shape=(nb_hinges, 2)
diff = np.zeros_like(r_orig_ic)
diff[1,0] += 0.8
diff[2,0] += 0.8
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0,1,2,2,3, 4,4,5,5,6, 10,8,9,11,6 ,10,11,7,14,13,  14,16,18,19,17, 15,16,17])
j_p = np.array([4,4,1,5,5, 6,8,9,7,10, 8,9,11,7,14 ,12,13,15,12,15, 16,12,12,13,13, 17,18,19])
# angles
i_t = np.array([1,0,6,8,9,5,2,1,4,3,7,9,10,8,4,14,12,10,6,16,12,14,18,12,16,18,11,17,17,13,15,13,17,7,11,13,11,7,5,8,10,15])  # containing first end point
j_t = np.array([4,4,4,4,8,9,5,2,1,5,5,5,6,10,8,6,14,12,10,14,16,12,16,18,12,12,13,13,19,17,17,15,13,15,7,11,9,11,7,9,8,13])  # containing angle points2
k_t = np.array([0,6,8,1,4,8,9,5,2,2,3,7,4,6,10,10,6,14,12,12,14,16,12,16,18,10,19,19,13,19,13,17,15,13,15,7,5,9,11,11,9,11])  # containing second end point
