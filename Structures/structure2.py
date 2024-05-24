import numpy as np

"""
            2d structure
           0---1----2----3---4
                \  / \  /
                5 6   7 8
                /  \ /  \
           9---10---11---12--13
    """
# positions
r_orig_ic = np.array(
    [[0, 1], [1, 1], [2.5, 1], [4, 1], [5, 1], [1.5, 0.5], [2, 0.5], [3, 0.5], [3.5, 0.5], [0, 0], [1, 0], [2.5, 0],
     [4, 0], [5, 0]], dtype=float)  # shape=(nb_positions, 2)
diff = np.zeros_like(r_orig_ic)
diff[4, 0] += 1
diff[13, 0] += 1
r_stressed_ic = r_orig_ic + diff
# pairs
i_p = np.array([0, 1, 2, 3, 1, 2, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
j_p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 11, 12, 10, 11, 12, 13])
# angles
i_t = np.array([0, 5, 1, 6, 7, 2, 8, 1, 2, 2, 3, 5, 11, 6, 7, 12, 8, 13])  # containing first end point
j_t = np.array([1, 1, 2, 2, 2, 3, 3, 5, 6, 7, 8, 10, 10, 11, 11, 11, 12, 12])  # containing angle points2
k_t = np.array([5, 2, 6, 7, 3, 8, 4, 10, 11, 11, 12, 9, 5, 10, 6, 7, 11, 8])  # containing second end point
