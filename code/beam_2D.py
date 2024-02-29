import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""  
2d structure
            2 
_/\_    0_1/ \3_5
 \/        \ /
        x   4
       0,0
            
suffix:
        _i: nb_hinge
        _n: nb_body
        _c: cartesian
        _alpha: beam related
        _beta: angle related
"""

"""
calculates the energy U of the overall system
"""
def UBeam_2D(positions_ic, beam_lengths_n, c_alpha, i_n, j_n):
    U = 0.0
    for m in range(len(i_n)):
        index_i = i_n[m]
        index_j = j_n[m]
        # print("\nm: ", m , "\npos j: ", positions_ic[index_j], "\npos i: ", positions_ic[index_i])
        U += 0.5 * c_alpha[m] * ((np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m]) ** 2)
    return U

"""
calculates the derivative of the function U for every position
"""
def dUBeam_2D(positions_ic, beam_lengths_n, c_alpha, i_i, j_i):
    dU = np.zeros([len(positions_ic),2])
    r_hat = getRHat_2D(positions_ic, i_i, j_i)
    for m in range(len(c_alpha)):
        index_j = j_i[m]
        index_i = i_i[m]
        dU[index_i] += c_alpha[m] * (np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m]) * r_hat[m]
        dU[index_j] -= c_alpha[m] * (np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m]) * r_hat[m]
    return dU


"""
calculates the energy U of the hinges
U = 0.5 * c_beta_i * (cos(theta_ijk) - cos(theta_0))²
cos(theta_ijk) = (r_ji * r_jk)/(|r_ij| * |r_jk|) 
"""
def UAngle_2D(positions_initial_ic, positions_final_ic, c_beta, i_beta, j_beta, k_beta):
    U = 0.0
    cos_angles_ijk = getCosAngles_2D(positions_final_ic, i_beta, j_beta, k_beta)
    cos_angles_0 = getCosAngles_2D(positions_initial_ic, i_beta, j_beta, k_beta)
    for m in range(len(c_beta)):
        U += 0.5 * c_beta[m] * (cos_angles_ijk[m] - cos_angles_0[m])**2
    return U


def dUAngle_2D(positions_inital_ic, positions_final_ic, c_beta_i, i_beta, j_beta, k_beta):
    dU = np.zeros([len(c_beta_i)])

    cos_angles_0 = getCosAngles_2D(positions_inital_ic, i_beta, j_beta, k_beta)
    cos_angles_ijk = getCosAngles_2D(positions_final_ic, i_beta, j_beta, k_beta)

    # loop over all positions
    for m in range(len(positions_inital_ic)):
        index_i = i_beta[m]
        index_j = j_beta[m]
        index_k = k_beta[m]
        for n in range(len(i_beta)):
            dU[m] = c_beta_i[m] * (cos_angles_ijk[m] - cos_angles_0[m]) *
    return dU

def UTriplet_2D(positions_inital_ic, positions_final_ic, c_alpha, i_gamma, j_gamma, k_gamma):
    U = 0.0
    for m in range(len(i_gamma)):
        index_i = i_gamma[m]
        index_j = j_gamma[m]
        index_k = k_gamma[m]

        beamlengths_ij_0 = getBeamLength_2D(positions_inital_ic, i_gamma, j_gamma)
        beamlengths_jk_0 = getBeamLength_2D(positions_inital_ic, j_gamma, k_gamma)
        beamlengths_ki_0 = getBeamLength_2D(positions_inital_ic, k_gamma, i_gamma)

        # compute the energy of every beam in the triplet
        V_ij = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_j] - positions_final_ic[index_i]) - beamlengths_ij_0[m]) ** 2)
        V_jk = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_k] - positions_final_ic[index_j]) - beamlengths_jk_0[m]) ** 2)
        V_ki = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_i] - positions_final_ic[index_k]) - beamlengths_ki_0[m]) ** 2)
        # add all beam energies up to overall triplet energy and sum all triplet energies
        U += V_ij + V_jk + V_ki
    return U


def dUTriplet_2D(positions_inital_ic, positions_final_ic, c_alpha, i_gamma, j_gamma, k_gamma):
    dU = np.zeros([len(positions_inital_ic), 2])
    for m in range(len(i_gamma)):
        index_i = i_gamma[m]
        index_j = j_gamma[m]
        index_k = k_gamma[m]

        beamlengths_ij_0 = getBeamLength_2D(positions_inital_ic, i_gamma, j_gamma)
        beamlengths_jk_0 = getBeamLength_2D(positions_inital_ic, j_gamma, k_gamma)
        beamlengths_ki_0 = getBeamLength_2D(positions_inital_ic, k_gamma, i_gamma)

        r_hat_ij = getRHat_2D(positions_final_ic, i_gamma, j_gamma)
        r_hat_jk = getRHat_2D(positions_final_ic, j_gamma, k_gamma)
        r_hat_ki = getRHat_2D(positions_final_ic, k_gamma, i_gamma)

        dV_ij = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_j] - positions_final_ic[index_i]) - beamlengths_ij_0[m]) ** 2) * r_hat_ij[m]
        dV_jk = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_k] - positions_final_ic[index_j]) - beamlengths_jk_0[m]) ** 2) * r_hat_jk[m]
        dV_ki = 0.5 * c_alpha[m] * ((np.linalg.norm(positions_final_ic[index_i] - positions_final_ic[index_k]) - beamlengths_ki_0[m]) ** 2) * r_hat_ki[m]
        dU[m] = dV_ij + dV_jk + dV_ki

    return dU


"""
returns the normalized vector for every connection of positions
"""
def getRHat_2D(positions_ic, i_alpha, j_alpha):
    r_hat = np.zeros([len(i_alpha), 2])
    for m in range(len(i_alpha)):
        index_i = i_alpha[m]
        index_j = j_alpha[m]
        r_hat[m] = ((positions_ic[index_i] - positions_ic[index_j]) / np.linalg.norm(positions_ic[index_i] - positions_ic[index_j]))
    return r_hat

"""
returns an array containing the beamlength of every beam, index is the number of the beam
"""
def getBeamLength_2D(positions_ic, i_alpha, j_alpha):
    # beamlengths of bodies (_n for bodies)
    # calculate beam_length: sqrt((x_n+1 - x_n)² + (y_n+1 - y_n)²)
    # 1: np.diff(positions_ic, axis=0), 2.np.linalg.norm(position_diff_ic, axis=0)
    beam_lengths_n = np.zeros(len(i_alpha))
    for m in range(len(i_alpha)):
        index_i = i_alpha[m]
        index_j = j_alpha[m]
        beam_lengths_n[m] = np.linalg.norm(positions_ic[index_j] - positions_ic[index_i])
    return beam_lengths_n

def getCosAngles_2D(positions_ic, i_beta, j_beta, k_beta):
    angles = np.zeros(len(i_beta), dtype=float)
    for m in range(len(i_beta)):
        index_i = i_beta[m]
        index_j = j_beta[m]
        index_k = k_beta[m]
        r_ij = positions_ic[index_j] - positions_ic[index_i]
        r_kj = positions_ic[index_j] - positions_ic[index_k]
        nominator = np.dot((r_ij), (r_kj))
        denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)
        # print("nom: ", nominator, "\ndom: ", denominator)
        angles[m] = nominator/denominator
    return angles

# def getSinAngles_2D(positions_ic, i_beta, j_beta, k_beta):
#     angles = np.zeros(len(i_beta), dtype=float)
#     for m in range(len(i_beta)):
#         index_i = i_beta[m]
#         index_j = j_beta[m]
#         index_k = k_beta[m]
#         r_ij = positions_ic[index_j] - positions_ic[index_i]
#         r_kj = positions_ic[index_j] - positions_ic[index_k]
#         nominator = np.absolute(np.cross(r_ij, r_kj))
#         denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)
#         # print("nom: ", nominator, "\ndom: ", denominator)
#         angles[m] = nominator/denominator
#     return angles

"""
objective U function for using in the optimizer.
border constraints are defined by con1 and con2
"""
def dUBeamObjective_2D(positions_flat, beam_lengths_n, c_alpha, i_alpha, j_alpha, con1, con2):
    positions_ic = positions_flat.reshape(nb_hinges-2, 2)
    U = 0.0
    for m in range(len(i_alpha)):
        if m == 0:
            index_j = j_alpha[m] - 1
            U += 0.5 * c_alpha[m] * ((np.linalg.norm(positions_ic[index_j] - con1) - beam_lengths_n[m]) ** 2)
        elif m == (len(i_alpha) - 1):
            index_i = i_alpha[m] - 1
            U += 0.5 * c_alpha[m] * ((np.linalg.norm(con2 - positions_ic[index_i]) - beam_lengths_n[m]) ** 2)
        else:
            index_i = i_alpha[m] - 1
            index_j = j_alpha[m] - 1
            U += 0.5 * c_alpha[m] * ((np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m]) ** 2)
    return U


if __name__ == "__main__":
    # beam
    i_alpha = np.array([0, 1, 2, 1, 4, 3])
    j_alpha = np.array([1, 2, 3, 4, 3, 5])

    # angles
    i_beta = np.array([0, 2, 4, 3, 1, 4, 2, 5]) # containing first end point
    j_beta = np.array([1, 1, 1, 2, 4, 3, 3, 3]) # containing angle points
    k_beta = np.array([2, 4, 0, 1, 3, 2, 5, 4]) # containing second end point

    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3.5, 1], [2, 0], [5, 1]])

    nb_bodies = len(i_alpha)
    nb_hinges = len(positions_initial_ic)
    nb_angles = len(i_beta)

    positions_flat = positions_final_ic[1:-1]
    positions_flat = positions_flat.reshape((nb_hinges-2) * 2)
    beam_lengths_n = getBeamLength_2D(positions_initial_ic, i_alpha, j_alpha)
    c_alpha = np.full(nb_bodies, 10)
    c_beta = np.ones(nb_angles)

    # con1 = positions_initial_ic[0]
    # con2 = positions_final_ic[-1]
    # res = scipy.optimize.minimize(objective_beam_2D, x0=positions_flat, args=(beam_lengths_n, c_alpha, i_alpha, j_alpha, con1, con2)) #, constraints=cons)#,jac=dU_2D)
    # points = res.x
    # points = np.insert(points, 0, con1)
    # points = np.append(points, con2)
    # points = points.reshape(nb_hinges,2)
    # plt.subplot(111, aspect=1)
    # for i, j in zip(i_alpha, j_alpha):
    #     plt.plot([positions_initial_ic[i][0], positions_initial_ic[j][0]], [positions_initial_ic[i][1], positions_initial_ic[j][1]], 'ob--')
    #     plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'xk-')
    # plt.show()


    # positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    # positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3.5, 1], [2, 0], [5, 1]])
    # beam calc
    # cos_angles_0 = getCosAngles_2D(positions_initial_ic, i_beta, j_beta, k_beta)
    # cos_angles_ijk = getCosAngles_2D(positions_final_ic, i_beta, j_beta, k_beta)
    #
    # # sin_angles_0 = getSinAngles_2D(positions_initial_ic, i_beta, j_beta, k_beta)
    # # sin_angles_ijk = getSinAngles_2D(positions_final_ic, i_beta, j_beta, k_beta)
    #
    # print("\ncos_0: ", cos_angles_0, "\n\ncos_ijk: ", cos_angles_ijk)
    # print("\nsin_0: ", sin_angles_0, "\n\nsin_ijk: ", sin_angles_ijk)
    #
    # print("\nU: ", UAngle_2D(positions_initial_ic, positions_final_ic, c_beta, i_beta, j_beta, k_beta))
    # print("\ndU: ", dUAngle_2D(positions_initial_ic, positions_final_ic, c_beta, i_beta, j_beta, k_beta))


