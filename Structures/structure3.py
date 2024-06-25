import numpy as np
"""
            2d structure
           0--1---2--3
            \ /\ /\ /
             4  5  6
            / \/ \/ \
           7--8---9--10
    """

# positions
r_orig_ic = np.array([[0, 2], [1, 2], [2, 2], [3, 2], [0.5, 1], [1.5, 1], [2.5, 1], [0, 0], [1, 0], [2, 0], [3, 0]],
                     dtype=float)  # shape=(nb_positions, 2)
posDisplaced = np.array[[3,10]]
dimDisplaced = 0
distanceDisplaced = 1
diff = np.zeros_like(r_orig_ic)
diff[posDisplaced, dimDisplaced] += distanceDisplaced
r_stressed_ic = r_orig_ic + diff
# # pairs
i_p = np.array([0, 1, 2, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9])
j_p = np.array([1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9, 10, 8, 9, 10])
# angles
i_a_t = np.array(
    [1, 4, 0, 5, 2, 5, 1, 6, 3, 6, 2, 7, 1, 8, 2, 9, 3, 8, 9, 10, 4, 7, 4, 5, 8, 5, 6, 9])  # containing first end point
j_a_t = np.array(
    [0, 1, 4, 1, 1, 2, 5, 2, 2, 3, 6, 4, 4, 5, 5, 6, 6, 4, 5, 6, 7, 8, 8, 8, 9, 9, 9, 10])  # containing angle points2
k_a_t = np.array([4, 0, 1, 4, 5, 1, 2, 5, 6, 2, 3, 0, 8, 1, 9, 2, 10, 7, 8, 9, 8, 4, 5, 9, 5, 6, 10,
                6])  # containing second end point
# triplets
i_t = np.array([0, 1, 2, 7, 8, 9])  # containing first end point
j_t = np.array([1, 2, 3, 4, 5, 6])  # containing angle points2
k_t = np.array([4, 5, 6, 8, 9, 10])  # containing second end point
