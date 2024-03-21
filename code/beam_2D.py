import numpy as np
from matscipy import numpy_tricks as nt
import scipy.optimize
import matplotlib.pyplot as plt

"""  
suffix:
        _i: position length
        _p: pair length
        _t: triplet length
        _c: cartesian
"""

class Beam:
    def __init__(self,  c_p, i_p, j_p):
        self.c_p = c_p
        self.i_p = i_p
        self.j_p = j_p

    """
    calculates the energy U of the overall system
    """
    def UBeam(self, r_ic, beamlengths_p):
        rij_pc = r_ic[self.j_p] - r_ic[self.i_p]  # vector rij
        rij_p = np.linalg.norm(rij_pc, axis=1)  # length of vector rij
        return np.sum(0.5 * self.c_p * (rij_p - beamlengths_p) ** 2)

    """
    calculates the derivative of the function U for every position
    """
    def dUBeam(self, r_ic, beamlengths_p):
        rij_pc = r_ic[self.i_p] - r_ic[self.j_p]
        rij_p = np.linalg.norm(rij_pc, axis=1)
        rijHat_pc = (rij_pc.T / rij_p).T

        dU_p = nt.mabincount(self.i_p, (self.c_p * (rij_p - beamlengths_p) * rijHat_pc.T).T, len(r_ic), axis=0)
        dU_p -= nt.mabincount(self.j_p, (self.c_p * (rij_p - beamlengths_p) * rijHat_pc.T).T, len(r_ic), axis=0)
        return dU_p



class Angle:
    def __init__(self, c_t, i_t, j_t, k_t):
        self.c_t = c_t
        self.i_t = i_t
        self.j_t = j_t
        self.k_t = k_t

    """
    calculates the energy U of the hinges
    U = 0.5 * c_beta_i * (cos(theta_ijk) - cos(theta_0))²
    cos(theta_ijk) = (r_ji * r_jk)/(|r_ij| * |r_jk|) 
    """
    def UAngle(self, cosijk_t, cos0_t):
        return np.sum(0.5 * self.c_t * (cosijk_t - cos0_t) ** 2)


    def dUAngle(self, r_ic, cosijk_t, cos0_t):
        rij_tc = r_ic[self.i_t] - r_ic[self.j_t]
        rij_t = np.linalg.norm(rij_tc, axis=1)
        rijHat_tc = (rij_tc.T / rij_t).T
        rkj_tc = r_ic[self.k_t] - r_ic[self.j_t]
        rkj_t = np.linalg.norm(rkj_tc, axis=1)
        rkjHat_tc = (rkj_tc.T / rkj_t).T
        rik_tc = r_ic[self.i_t] - r_ic[self.k_t]
        rik_t = np.linalg.norm(rik_tc, axis=1)
        #rikHat_tc = (rik_tc.T / rik_t).T

        # print("\ni:", i_t.shape)
        # print("\nshape: ", ((c_t * (cosijk_t - cos0_t) * (rkjHat_tc - (cosijk_t * rijHat_tc.T).T).T) / rij_t).shape)

        dU_ci = nt.mabincount(self.i_t,
                              self.c_t * (cosijk_t - cos0_t) * (rkjHat_tc.T - (cosijk_t * rijHat_tc.T)) / rij_t,
                              minlength=r_ic.shape[0],
                              axis=1)
        dU_ci -= nt.mabincount(self.j_t,
                               self.c_t * (cosijk_t - cos0_t) * (rkjHat_tc.T - (cosijk_t * rijHat_tc.T)) / rij_t,
                               minlength=r_ic.shape[0],
                               axis=1)
        dU_ci += nt.mabincount(self.k_t,
                               self.c_t * (cosijk_t - cos0_t) * (rijHat_tc.T - (cosijk_t * rkjHat_tc.T)) / rkj_t,
                               minlength=r_ic.shape[0],
                               axis=1)
        dU_ci -= nt.mabincount(self.j_t,
                               self.c_t * (cosijk_t - cos0_t) * (rijHat_tc.T - (cosijk_t * rkjHat_tc.T)) / rkj_t,
                               minlength=r_ic.shape[0],
                               axis=1)
        return dU_ci.T


class Triplet:
    def __init__(self, c_t3, i_t, j_t, k_t):
        self.c_t3 = c_t3
        self.i_t = i_t
        self.j_t = j_t
        self.k_t = k_t

    def UTriplet(self, r_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t):
        rij_tc = r_ic[self.i_t] - r_ic[self.j_t]
        rij_t = np.linalg.norm(rij_tc, axis=1)
        rkj_tc = r_ic[self.k_t] - r_ic[self.j_t]
        rkj_t = np.linalg.norm(rkj_tc, axis=1)
        rik_tc = r_ic[self.i_t] - r_ic[self.k_t]
        rik_t = np.linalg.norm(rik_tc, axis=1)
        cij_t = self.c_t3[:, 0]
        ckj_t = self.c_t3[:, 1]
        cik_t = self.c_t3[:, 2]

        # compute the energy of every beam in the triplet
        V_t = 0.5 * cij_t * ((rij_t - beamlengths0ij_t) ** 2)
        V_t += 0.5 * ckj_t * ((rkj_t - beamlengths0kj_t) ** 2)
        V_t += 0.5 * cik_t * ((rik_t - beamlengths0ik_t) ** 2)
        # print("\n",V_ij,"\n", V_kj,"\n", V_ki)
        # add all beam energies up to overall triplet energy and sum all triplet energies
        return np.sum(V_t)


    def dUTriplet(self, r_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t):
        rij_tc = r_ic[self.i_t] - r_ic[self.j_t]
        rij_t = np.linalg.norm(rij_tc, axis=1)
        rijHat_tc = (rij_tc.T / rij_t).T

        rkj_tc = r_ic[self.k_t] - r_ic[self.j_t]
        rkj_t = np.linalg.norm(rkj_tc, axis=1)
        rkjHat_tc = -(rkj_tc.T / rkj_t).T

        rik_tc = r_ic[self.i_t] - r_ic[self.k_t]
        rik_t = np.linalg.norm(rik_tc, axis=1)
        rikHat_tc = (rik_tc.T / rik_t).T

        cij_t = self.c_t3[:, 0]
        ckj_t = self.c_t3[:, 1]
        cik_t = self.c_t3[:, 2]

        dU_tc = nt.mabincount(self.i_t,
                              (cij_t * ((rij_t - beamlengths0ij_t)) * rijHat_tc.T),
                              len(r_ic),
                              axis=1)
        dU_tc += nt.mabincount(self.j_t,
                               (cij_t * ((rij_t - beamlengths0ij_t)) * rijHat_tc.T),
                               len(r_ic),
                               axis=1)
        dU_tc -= nt.mabincount(self.k_t,
                               (ckj_t * ((rkj_t - beamlengths0kj_t)) * rkjHat_tc.T),
                               len(r_ic),
                               axis=1)
        dU_tc += nt.mabincount(self.j_t,
                               (ckj_t * ((rkj_t - beamlengths0kj_t)) * rkjHat_tc.T),
                               len(r_ic),
                               axis=1)
        dU_tc -= nt.mabincount(self.k_t,
                               (cik_t * ((rik_t - beamlengths0ik_t)) * rikHat_tc.T),
                               len(r_ic),
                               axis=1)
        dU_tc += nt.mabincount(self.i_t,
                               (cik_t * ((rik_t - beamlengths0ik_t)) * rikHat_tc.T),
                               len(r_ic),
                               axis=1)

        return dU_tc.T


"""
returns an array containing the beamlength of every beam, index is the number of the beam
"""
def getBeamLength(r_ic, i_p, j_p):
    beamlengths_p = np.linalg.norm(r_ic[i_p] - r_ic[j_p], axis=1)
    return beamlengths_p


def getRHat(r_ic, i_p, j_p):
    rij_pc = r_ic[i_p] - r_ic[j_p]
    rij_p = np.linalg.norm(rij_pc, axis=1)
    rijHat = rij_pc / rij_p
    return rijHat


def getCosAngles(r_ic, i_t, j_t, k_t):
    rij_tc = r_ic[i_t] - r_ic[j_t]
    rkj_tc = r_ic[k_t] - r_ic[j_t]
    # print("\nrij: ", r_ij, "\nrkj: ", r_kj)
    nominator = np.sum(rij_tc * rkj_tc, axis=1)
    #     print("\nnominator: ",nominator)
    denominator = np.linalg.norm(rij_tc, axis=1) * np.linalg.norm(rkj_tc, axis=1)
    # print("nom: ", nominator, "\ndom: ", denominator)
    angles = nominator / denominator
    return angles


# def getSinAngles(r_ic, i_t, j_t, k_t):
#     angles = np.zeros(len(i_t), dtype=float)
#     for m in range(len(i_t)):
#         index_i = i_t[m]
#         index_j = j_t[m]
#         index_k = k_t[m]
#         r_ij = r_ic[index_j] - r_ic[index_i]
#         r_kj = r_ic[index_j] - r_ic[index_k]
#         nominator = np.absolute(np.cross(r_ij, r_kj))
#         denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)
#         # print("nom: ", nominator, "\ndom: ", denominator)
#         angles[m] = nominator/denominator
#     return angles

"""
objective U function for using in the optimizer.
border constraints are defined by con1 and con2
"""
def dUBeamObjective(positions_flat, beamlengths_p, c_p, i_p, j_p, con1, con2):
    r_ic = positions_flat.reshape(nb_hinges - 2, 2)
    U = 0.0
    for m in range(len(i_p)):
        if m == 0:
            index_j = j_p[m] - 1
            U += 0.5 * c_p[m] * ((np.linalg.norm(r_ic[index_j] - con1) - beamlengths_p[m]) ** 2)
        elif m == (len(i_p) - 1):
            index_i = i_p[m] - 1
            U += 0.5 * c_p[m] * ((np.linalg.norm(con2 - r_ic[index_i]) - beamlengths_p[m]) ** 2)
        else:
            index_i = i_p[m] - 1
            index_j = j_p[m] - 1
            U += 0.5 * c_p[m] * ((np.linalg.norm(r_ic[index_j] - r_ic[index_i]) - beamlengths_p[m]) ** 2)
    return U


if __name__ == "__main__":
    """
        2d structure
                    2 
        _/\_    0_1/ \3_5
         \/        \ /
                x   4
               0,0
    """

    # beam
    i_p = np.array([0, 1, 2, 1, 4, 3])
    j_p = np.array([1, 2, 3, 4, 3, 5])

    # angles
    i_t = np.array([0, 2, 4, 3, 1, 4, 2, 5])  # containing first end point
    j_t = np.array([1, 1, 1, 2, 4, 3, 3, 3])  # containing angle points
    k_t = np.array([2, 4, 0, 1, 3, 2, 5, 4])  # containing second end point

    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3.5, 1], [2, 0], [5, 1]])

    nb_bodies = len(i_p)
    nb_hinges = len(positions_initial_ic)
    nb_angles = len(i_t)

    positions_flat = positions_final_ic[1:-1]
    positions_flat = positions_flat.reshape((nb_hinges - 2) * 2)
    beamlengths_p = getBeamLength(positions_initial_ic, i_p, j_p)
    c_p = np.full(nb_bodies, 10)
    c_t = np.ones(nb_angles)

    UBeam(positions_final_ic, beamlengths_p, c_p, i_p, j_p)
    dUBeam(positions_final_ic, beamlengths_p, c_p, i_p, j_p)

    # con1 = positions_initial_ic[0]
    # con2 = positions_final_ic[-1]
    # res = scipy.optimize.minimize(objective_beam, x0=positions_flat, args=(beamlengths_p, c_p, i_p, j_p, con1, con2)) #, constraints=cons)#,jac=dU)
    # points = res.x
    # points = np.insert(points, 0, con1)
    # points = np.append(points, con2)
    # points = points.reshape(nb_hinges,2)
    # plt.subplot(111, aspect=1)
    # for i, j in zip(i_p, j_p):
    #     plt.plot([positions_initial_ic[i][0], positions_initial_ic[j][0]], [positions_initial_ic[i][1], positions_initial_ic[j][1]], 'ob--')
    #     plt.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]], 'xk-')
    # plt.show()

    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3.5, 1], [2, 0], [5, 1]])
    # beam calc
    cos0_t = getCosAngles(positions_initial_ic, i_t, j_t, k_t)
    cosijk_t = getCosAngles(positions_final_ic, i_t, j_t, k_t)

    print("\nU: ", UAngle(cosijk_t, cos0_t, c_t))
    print("\ndU: ", dUAngle(positions_final_ic, cosijk_t, cos0_t, c_t, i_t, j_t, k_t))
