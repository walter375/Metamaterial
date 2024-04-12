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
    def __init__(self, c_p, i_p, j_p):
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

    def UBeamObjective(self, ric_flat, beamlengths_p, borderCon, r_stressed_ic):
        r_ic = ric_flat.reshape(len(ric_flat)//2, 2)
        for i in range(len(borderCon)):
            r_ic = np.insert(r_ic, borderCon[i], r_stressed_ic[borderCon[i]], axis=0)
        rij_pc = r_ic[self.j_p] - r_ic[self.i_p]  # vector rij
        rij_p = np.linalg.norm(rij_pc, axis=1)  # length of vector rij
        return np.sum(0.5 * self.c_p * (rij_p - beamlengths_p) ** 2)

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
        # rikHat_tc = (rik_tc.T / rik_t).T

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

    def UAngleObjective(self, ric_flat, cos0_t, borderCon, r_stressed_ic):
        r_ic = ric_flat.reshape(len(ric_flat) // 2, 2)
        for i in range(len(borderCon)):
            r_ic = np.insert(r_ic, borderCon[i], r_stressed_ic[borderCon[i]], axis=0)
        cosijk_t = getCosAngles(r_ic, self.i_t, self.j_t, self.k_t)
        return np.sum(0.5 * self.c_t * (cosijk_t - cos0_t) ** 2)


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
#todo test
def getBeamLength(r_ic, i_p, j_p):
    beamlengths_p = np.linalg.norm(r_ic[i_p] - r_ic[j_p], axis=1)
    return beamlengths_p

#todo test
def getRHat(r_ic, i_p, j_p):
    rij_pc = r_ic[i_p] - r_ic[j_p]
    rij_p = np.linalg.norm(rij_pc, axis=1)
    rijHat = rij_pc / rij_p
    return rijHat

#todo test
def getCosAngles(r_ic, i_t, j_t, k_t):
    rij_tc = r_ic[i_t] - r_ic[j_t]
    rkj_tc = r_ic[k_t] - r_ic[j_t]
    # print("\nrij: ", r_ij, "\nrkj: ", r_kj)
    nominator_t = np.sum(rij_tc * rkj_tc, axis=1)
    # print("\nnominator: ", nominator)
    denominator_t = np.linalg.norm(rij_tc, axis=1) * np.linalg.norm(rkj_tc, axis=1)
    angles_t = nominator_t / denominator_t
    return angles_t

#todo test
def getBorderPoints(r_ic):
    # upper bounds
    xmax, ymax = np.max(r_ic, axis=0)
    # x
    rows, cols = np.where(r_ic == xmax)
    right = rows[np.where(cols == 0)]
    # y
    rows, cols = np.where(r_ic == ymax)
    upper = rows[np.where(cols == 1)]
    # lower bounds
    xmin, ymin = np.min(r_ic, axis=0)
    # x
    rows, cols = np.where(r_ic == xmin)
    left = rows[np.where(cols == 0)]
    # y
    rows, cols = np.where(r_ic == ymin)
    lower = rows[np.where(cols == 1)]

    return left, right, lower, upper

def conLen(ric_flat):
    r_ic = ric_flat.reshape(len(ric_flat)//2, 2)
    for i in range(len(border)):
        r_ic = np.insert(r_ic, border[i], r_stressed_ic[border[i]], axis=0)
    len0 = getBeamLength(r_orig_ic, i_p, j_p)
    len1 = getBeamLength(r_ic, i_p, j_p)
    # print(len1-len0)
    return len1 - len0

"""
objective U function for using in the optimizer.
borderCon constraints are defined by con1 and con2
"""
if __name__ == "__main__":
    """
        2d structure
                    2 
        _/\_    0_1/ \3_5
         \/        \ /
                x   4
               0,0
    """

    ''' initialization '''
    # positions
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

    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    beam.UBeam(r_stressed_ic, beamlengths_p)
    beam.dUBeam(r_stressed_ic, beamlengths_p)
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' optimizer '''
    left, right, upper, lower = getBorderPoints(r_stressed_ic)
    border = np.concatenate((left, right), 0)
    border = np.sort(border)
    ric_flat = np.delete(r_stressed_ic, border, 0).flatten()
    # beams
    res = scipy.optimize.minimize(beam.UBeamObjective, x0=ric_flat, args=(beamlengths_p, border, r_stressed_ic))#, jac=beam.dUBeam)
    points1 = res.x
    points1 = points1.reshape(len(points1)//2, 2)
    for i in range(len(border)):
        points1 = np.insert(points1, border[i], r_stressed_ic[border[i]], axis=0)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=all, sharey=all)
    for i, j in zip(i_p, j_p):
        ax1.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax2.plot([points1[i][0], points1[j][0]], [points1[i][1], points1[j][1]], 'vy:')
        # ax1.plot([points1[i][0], points1[j][0]], [points1[i][1], points1[j][1]], 'xk-')
    # angles
    cons1 = [{'type': 'eq', 'fun': conLen}]
    res = scipy.optimize.minimize(angle.UAngleObjective, x0=ric_flat, args=(cos0_t, border, r_stressed_ic), constraints=cons1) #, jac=angle.dUAngle)
    points2 = res.x
    points2 = points2.reshape(len(points2) // 2, 2)
    for i in range(len(border)):
        points2 = np.insert(points2, border[i], r_stressed_ic[border[i]], axis=0)
    for i, j in zip(i_p, j_p):
        # ax1.plot([points2[i][0], points2[j][0]], [points2[i][1], points2[j][1]], 'xr--')
        # ax2.plot([points2[i][0], points2[j][0]], [points2[i][1], points2[j][1]], 'xr--')
        ax3.plot([points2[i][0], points2[j][0]], [points2[i][1], points2[j][1]], 'xr--')
    ax1.set_title("Inital system")
    ax2.set_title("Beam model optimization")
    ax3.set_title("Angle model optimization")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    """
            2d structure
           0---1----2----3---4
                \  / \  /
                5 6   7 8
                /  \ /  \
           9---10---11---12--13
    """

    ''' initialization '''
    # positions
    r_orig_ic = np.array([[0, 1], [1,1], [2.5,1], [4,1], [5,1], [1.5,0.5], [2,0.5], [3,0.5], [3.5,0.5], [0, 0], [1,0], [2.5,0], [4,0], [5,0]], dtype=float)  # shape=(nb_hinges, 2)
    diff = np.zeros_like(r_orig_ic)
    diff[4,0] += 0.5
    diff[13,0] += 0.5
    r_stressed_ic = r_orig_ic + diff
    # pairs
    i_p = np.array([0, 1, 2, 3, 1, 2, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12])
    j_p = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 11, 12, 10, 11, 12, 13])
    # angles
    i_t = np.array([0, 5, 1, 6, 7, 2, 8, 1, 2, 2, 3, 5, 11, 6, 7, 12, 8, 13])  # containing first end point
    j_t = np.array([1, 1, 2, 2, 2, 3, 3, 5, 6, 7, 8, 10, 10, 11, 11, 11, 12, 12])  # containing angle points2
    k_t = np.array([5, 2, 6, 7, 3, 8, 4, 10, 11, 11, 12, 9, 5, 10, 6, 7, 11, 8 ])  # containing second end point

    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]

    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    beam.UBeam(r_stressed_ic, beamlengths_p)
    beam.dUBeam(r_stressed_ic, beamlengths_p)

    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    left, right, upper, lower = getBorderPoints(r_orig_ic)

    ''' optimizer '''
    border = np.concatenate((left, right), 0)
    border = np.sort(border)
    ric_flat = np.delete(r_stressed_ic, border,0).flatten()

    # beams
    res = scipy.optimize.minimize(beam.UBeamObjective, x0=ric_flat, args=(beamlengths_p, border, r_stressed_ic))#, jac=beam.dUBeam)
    points3 = res.x
    points3 = points3.reshape(len(points3)//2, 2)
    for i in range(len(border)):
        points3 = np.insert(points3, border[i], r_stressed_ic[border[i]], axis=0)
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=all, sharey=all)
    for i, j in zip(i_p, j_p):
        ax1.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax2.plot([points3[i][0], points3[j][0]], [points3[i][1], points3[j][1]], 'vy:')

    # angles
    cons = [{'type': 'eq', 'fun': conLen}]
    res = scipy.optimize.minimize(angle.UAngleObjective, x0=ric_flat, args=(cos0_t, border, r_stressed_ic), constraints=cons)  # , jac=angle.dUAngle)
    points4 = res.x
    points4 = points4.reshape(len(points4)//2, 2)
    for i in range(len(border)):
        points4 = np.insert(points4, border[i], r_stressed_ic[border[i]], axis=0)
    print(points4)
    for i, j in zip(i_p, j_p):
        #ax2.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ob--')
        ax3.plot([points4[i][0], points4[j][0]], [points4[i][1], points4[j][1]], 'xr--')
    ax1.set_title("Inital system")
    ax2.set_title("Beam model optimization")
    ax3.set_title("Angle model optimization")
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    plt.show()


# def getSinAngles(r_orig_ic, i_t, j_t, k_t):
#     angles = np.zeros(len(i_t), dtype=float)
#     for m in range(len(i_t)):
#         index_i = i_t[m]
#         index_j = j_t[m]
#         index_k = k_t[m]
#         r_ij = r_orig_ic[index_j] - r_orig_ic[index_i]
#         r_kj = r_orig_ic[index_j] - r_orig_ic[index_k]
#         nominator = np.absolute(np.cross(r_ij, r_kj))
#         denominator = np.linalg.norm(r_ij) * np.linalg.norm(r_kj)
#         # print("nom: ", nominator, "\ndom: ", denominator)
#         angles[m] = nominator/denominator
#     return angles

#def conLeft(ric_flat):
#     temp1 = ric_flat[0:2]
#     return temp1 - ric_flat[0:2]
#
# def con2(ric_flat):
#     temp2 = ric_flat[-2:]
#     return temp2 - ric_flat[-2:]
#
