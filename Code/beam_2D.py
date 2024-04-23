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

    def UBeamObjective(self, ric_flat, beamlengths_p):
        r_ic = ricUnflat(ric_flat, r_stressed_ic, border, x, y)
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

    def UAngle(self,cosijk_t, cos0_t):
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

    def UAngleObjective(self, ric_flat, cos0_t):
        r_ic = ricUnflat(ric_flat, r_stressed_ic, border, x, y)
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

    def UTripletObjective(self, ric_flat, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t):
        r_ic = ricUnflat(ric_flat, r_stressed_ic, border, x, y)
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

class BeamAngle():
    def __init__(self, Beam, Angle):
        self.Beam = Beam
        self.Angle = Angle
    def UBeamAngle(self, Beam, Angle):
        U = Beam.UBeam(Beam.r_ic, Beam.beamlengths_p) + Angle.UAngle(Angle.cosijk_t, Angle.cos0_t)
        return U

    def dUBeamAngle(self, Beam, Angle):
        dU = Beam.dUBeam(Beam.r_ic, Beam.beamlengths_p) + Angle.dUAngle(Beam.r_ic, Angle.cosijk_t, Angle.cos0_t)
        return dU

    def UBeamAngleObjective(self, ric_flat, beamlengths_p, cos0_t):
        U = beam.UBeamObjective(ric_flat, beamlengths_p) \
            + angle.UAngleObjective(ric_flat, cos0_t)
        return U

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
def getBorderPoints(r_ic, left=1, right=1, lower=0, upper=0):
    # get indices of max and min points in r_ic,
    # xmax = right, ymax = upper, xmin = left, ymin = lower
    xmax, ymax = np.max(r_ic, axis=0)
    xmin, ymin = np.min(r_ic, axis=0)
    border = np.array([], dtype=int)
    if right == 1:
        rows, cols = np.where(r_ic == xmax)
        right = rows[np.where(cols == 0)]
        border = np.append(border, right)
    if upper == 1:
        rows, cols = np.where(r_ic == ymax)
        upper = rows[np.where(cols == 1)]
        border = np.append(border, upper)
    if left == 1:
        rows, cols = np.where(r_ic == xmin)
        left = rows[np.where(cols == 0)]
        border = np.append(border, left)
    if lower == 1:
        rows, cols = np.where(r_ic == ymin)
        lower = rows[np.where(cols == 1)]
        border = np.append(border, lower)
    border = np.sort(border)
    return border

""" 
returns a flattened array of r_ic with all border points/constraints removed,
it can be chosen if only x or y or both should be removed
"""
def ricFlat(border, x=0, y=0):
    # fix points in x and y direction
    if x == 1 and y == 1:
        ric_flat = np.delete(r_stressed_ic, border, 0).flatten()
    # fix points in x direction
    elif x == 1 and y == 0:
        ric_flat = r_stressed_ic.flatten()
        ric_flat = np.delete(ric_flat, border*2)
    # fix points in y direction
    elif x == 0 and y == 1:
        ric_flat = r_stressed_ic.flatten()
        ric_flat = np.delete(ric_flat, border*2+1)
    # no fixing
    else:
        ric_flat = r_stressed_ic.flatten()
    return ric_flat

def ricUnflat(ric_flat, r_stressed_ic, border, x=0, y=0):
    if x == 1 and y == 1:
        r_ic = ric_flat.reshape(len(ric_flat)//2, 2)
        for i in range(len(border)):
            r_ic = np.insert(r_ic, border[i], r_stressed_ic[border[i]], axis=0)
    # fix points in x direction
    elif x == 1 and y == 0:
        for i in range(len(border)):
            ric_flat = np.insert(ric_flat, border[i]*2, r_stressed_ic.flatten()[border[i]*2])
        r_ic = ric_flat.reshape(len(ric_flat)//2, 2)
    # fix points in y direction
    elif x == 0 and y == 1:
        for i in range(len(border)):
            ric_flat = np.insert(ric_flat, border[i]*2+1, r_stressed_ic.flatten()[border[i]*2+1])
        r_ic = ric_flat.reshape(len(ric_flat) // 2, 2)
    # no fixing
    else:
        r_ic = ric_flat.reshape(len(ric_flat)//2, 2)
    # print(r_ic)
    # print(r_stressed_ic)
    return r_ic


def conLen(ric_flat):
    r_ic = ricUnflat(ric_flat, r_stressed_ic, border, x, y)
    len0 = getBeamLength(r_orig_ic, i_p, j_p)
    len1 = getBeamLength(r_ic, i_p, j_p)
    # print(len1-len0)
    return len1 - len0

def runOptimizer(function, x0, arguments, cons={}):
    res = scipy.optimize.minimize(function, x0=x0, args=(arguments), constraints=cons)  # , jac=beam.dUBeam)
    points = ricUnflat(res.x, r_stressed_ic, border, x, y)
    return points
def plotResults(points1, points2, points3, points4):
    f, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, sharex=all, sharey=all)
    for i, j in zip(i_p, j_p):
        ax1.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax2.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax3.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax4.plot([r_orig_ic[i][0], r_orig_ic[j][0]], [r_orig_ic[i][1], r_orig_ic[j][1]], 'ok-')
        ax1.plot([points1[i][0], points1[j][0]], [points1[i][1], points1[j][1]], 'vy:')
        ax2.plot([points2[i][0], points2[j][0]], [points2[i][1], points2[j][1]], 'xr--')
        ax3.plot([points3[i][0], points3[j][0]], [points3[i][1], points3[j][1]], 'xb--')
        ax4.plot([points4[i][0], points4[j][0]], [points4[i][1], points4[j][1]], 'og--')
    ax1.set_title("Beam model") # "Angle+Beam, s_b=%i, s_a=%i" %(stiffness_beam, stiffness_angle))
    ax2.set_title("Angle model")
    ax3.set_title("Angle+Beam s_b=%i, s_a=%i" % (stiffness_beam1,stiffness_angle1))
    ax4.set_title("Angle+Beam, s_b=%i, s_a=%i" % (stiffness_beam2, stiffness_angle2))
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    plt.show()
"""
objective U function for using in the optimizer.
borderCon constraints are defined by con1 and con2
"""
if __name__ == "__main__":
    from Structures import structure1 as s1
    r_orig_ic = s1.r_orig_ic
    r_stressed_ic = s1.r_stressed_ic
    i_p = s1.i_p
    j_p = s1.j_p
    i_t = s1.i_t
    j_t = s1.j_t
    k_t = s1.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # modifications
    x = 1
    y = 1
    left = 1
    right = 1
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
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    # ''' optimizer '''
    # border = getBorderPoints(r_stressed_ic, left, right)
    # ric_flat = ricFlat(border, x, y)
    # p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    # p2 = runOptimizer(angle.UAngleObjective, ric_flat, (cos0_t))
    # angle.c_t = np.full(nb_angles, stiffness_angle)
    # p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # beam.c_p = np.full(nb_bodies, stiffness_beam)
    # p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
#    plotResults(p1, p2, p3, p4)

    from Structures import structure2 as s2
    r_orig_ic = s2.r_orig_ic
    r_stressed_ic = s2.r_stressed_ic
    i_p = s2.i_p
    j_p = s2.j_p
    i_t = s2.i_t
    j_t = s2.j_t
    k_t = s2.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # modifications
    x = 1
    y = 0
    left = 1
    right = 1
    ''' beams '''
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    # cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    # ric_flat = ricFlat(border, x, y)
    # p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    # p2 = runOptimizer(angle.UAngleObjective, ric_flat, (cos0_t))
    # angle.c_t = np.full(nb_angles, stiffness_angle)
    # p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # beam.c_p = np.full(nb_bodies, stiffness_beam)
    # p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
#    plotResults(p1, p2, p3, p4)

    from Structures import structure3 as s3
    r_orig_ic = s3.r_orig_ic
    r_stressed_ic = s3.r_stressed_ic
    i_p = s3.i_p
    j_p = s3.j_p
    i_a_t = s3.i_a_t
    j_a_t = s3.j_a_t
    k_a_t = s3.k_a_t
    i_t = s3.i_t
    j_t = s3.j_t
    k_t = s3.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_triplets = i_t.shape[0]
    nb_angles = i_a_t.shape[0]
    # modifications
    x = 1
    y = 0
    left = 1
    right = 1
    # stiffness_beam = 10
    ''' beams '''
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    ''' angles '''
    c_a_t = np.ones(nb_angles)
    angle = Angle(c_a_t, i_a_t, j_a_t, k_a_t)
    cos0_t = getCosAngles(r_orig_ic, i_a_t, j_a_t, k_a_t)
    # cosijk_t = getCosAngles(r_stressed_ic, i_a_t, j_a_t, k_a_t)
    ''' Triplet'''
    beamlengths_ij_t = getBeamLength(r_orig_ic, i_t, j_t)
    beamlengths_kj_t = getBeamLength(r_orig_ic, j_t, k_t)
    beamlengths_ik_t = getBeamLength(r_orig_ic, i_t, k_t)
    c_t3 = np.ones([nb_triplets, 3])
    triplet = Triplet(c_t3, i_t, j_t, k_t)
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    # ric_flat = ricFlat(border, x, y)
    # p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    # p2 = runOptimizer(triplet.UTripletObjective, ric_flat, (beamlengths_ij_t, beamlengths_kj_t, beamlengths_ik_t))
    # angle.c_t = np.full(nb_angles, stiffness_angle)
    # p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # beam.c_p = np.full(nb_bodies, stiffness_beam)
    # p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # plotResults(p1, p2, p3, p4)

    from Structures import InverterMechanism as im
    r_orig_ic = im.r_orig_ic
    r_stressed_ic = im.r_stressed_ic
    i_p = im.i_p
    j_p = im.j_p
    i_t = im.i_t
    j_t = im.j_t
    k_t = im.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # constrains
    x = 1
    y = 1
    left = 1
    right = 0
    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    #cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    border = np.sort(np.append(border, 3)) # add point 3 to points to be fixed
    ric_flat = ricFlat(border, x, y)
    # p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    # cons = [{'type': 'eq', 'fun': conLen}]
    # p2 = runOptimizer(angle.UAngleObjective, ric_flat, (cos0_t), cons)
    # stiffness_angle = 10
    # angle.c_t = np.full(nb_angles, stiffness_angle)
    # stiffness_beam = 50
    # beam.c_t = np.full(nb_bodies, stiffness_beam)
    # p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # stiffness_beam = 200
    # stiffness_angle = 10
    # angle.c_t = np.full(nb_angles, stiffness_angle)
    # beam.c_p = np.full(nb_bodies, stiffness_beam)
    # p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # plotResults(p1,p2,p3,p4)

    from Structures import gripperWithHinges as g
    r_orig_ic = g.r_orig_ic
    r_stressed_ic = g.r_stressed_ic
    i_p = g.i_p
    j_p = g.j_p
    i_t = g.i_t
    j_t = g.j_t
    k_t = g.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # constrains
    x = 1
    y = 1
    left = 1
    right = 0
    # stiffness_beam = 50
    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    # cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    border = np.sort(np.append(border,6))  # add point 3 to points to be fixed
    ric_flat = ricFlat(border, x, y)
    beam.c_p[0] = 1000
    beam.c_p[7] = 1000
    beam.c_p[10] = 1000
    beam.c_p[13] = 1000
    beam.c_p[9] = 1000
    beam.c_p[12] = 1000
    p1 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    cons = [{'type': 'eq', 'fun': conLen}]
    p2 = runOptimizer(angle.UAngleObjective, ric_flat, cos0_t, cons)
    stiffness_beam1 = 100
    stiffness_angle1 = 10
    angle.c_t = np.full(nb_angles, stiffness_angle1)
    beam.c_p = np.full(nb_bodies, stiffness_beam1)
    p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    stiffness_beam2 = 500
    stiffness_angle2 = 10
    angle.c_t = np.full(nb_angles, stiffness_angle2)
    beam.c_p = np.full(nb_bodies, stiffness_beam2)
    p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    plotResults(p1, p2, p3, p4)

    from Structures import Auxetic as g
    r_orig_ic = g.r_orig_ic
    r_stressed_ic = g.r_stressed_ic
    i_p = g.i_p
    j_p = g.j_p
    i_t = g.i_t
    j_t = g.j_t
    k_t = g.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # constrains
    x = 1
    y = 1
    left = 1
    right = 1
    # stiffness_beam = 50
    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    # cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' beam angle combination '''
    # ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    ric_flat = ricFlat(border, x, y)
    # p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    # cons = [{'type': 'eq', 'fun': conLen}]
    # p2 = runOptimizer(angle.UAngleObjective, ric_flat, cos0_t, cons)
    # stiffness_angle1 = 10
    # stiffness_beam1 = 100
    # angle.c_t = np.full(nb_angles, stiffness_angle1)
    # beam.c_p = np.full(nb_bodies, stiffness_beam1)
    # p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # stiffness_angle2 = 10
    # stiffness_beam2 = 500
    # angle.c_t = np.full(nb_angles, stiffness_angle2)
    # beam.c_p = np.full(nb_bodies, stiffness_beam2)
    # p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    # plotResults(p1, p2, p3, p4)

    from Structures import gripperWithoutHinges as g
    r_orig_ic = g.r_orig_ic
    r_stressed_ic = g.r_stressed_ic
    i_p = g.i_p
    j_p = g.j_p
    i_t = g.i_t
    j_t = g.j_t
    k_t = g.k_t
    nb_bodies = i_p.shape[0]
    nb_hinges = r_orig_ic.shape[0]
    nb_angles = i_t.shape[0]
    # constrains
    x = 1
    y = 1
    left = 1
    right = 1
    # stiffness_beam = 200
    # stiffness_angle = 1
    ''' beams '''
    beamlengths_p = getBeamLength(r_orig_ic, i_p, j_p)
    c_p = np.ones(nb_bodies)
    beam = Beam(c_p, i_p, j_p)
    # beam.c_p[10:13] = beam.c_p[10:13] * 1000
    # beam.c_p[15:17] = beam.c_p[15:17] * 1000
    # beam.c_p[0:5] = beam.c_p[0:5] * 500
    ''' angles '''
    c_t = np.ones(nb_angles)
    angle = Angle(c_t, i_t, j_t, k_t)
    cos0_t = getCosAngles(r_orig_ic, i_t, j_t, k_t)
    cosijk_t = getCosAngles(r_stressed_ic, i_t, j_t, k_t)
    ''' beam angle combination '''
    ba = BeamAngle(beam, angle)
    ''' optimizer '''
    border = getBorderPoints(r_stressed_ic, left, right)
    border = np.sort(np.append(border,(1)))
    ric_flat = ricFlat(border, x, y)
    p1 = runOptimizer(beam.UBeamObjective, ric_flat, beamlengths_p)
    cons = [{'type': 'eq', 'fun': conLen}]
    p2 = runOptimizer(angle.UAngleObjective, ric_flat, (cos0_t), cons)

    stiffness_angle1 = 1
    stiffness_beam1 = 10
    angle.c_t = np.full(nb_angles, stiffness_angle1)
    beam.c_p = np.full(nb_bodies, stiffness_beam1)
    beam.c_p[9:11] = beam.c_p[9:11] * 100
    # beam.c_p[15:17] = beam.c_p[15:17] * 10
    # beam.c_p[0:5] = beam.c_p[0:5] * 10
    p3 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    stiffness_angle2 = 10
    stiffness_beam2 = 20
    angle.c_t = np.full(nb_angles, stiffness_angle2)
    beam.c_p = np.full(nb_bodies, stiffness_beam2)
    # beam.c_p[10:13] = beam.c_p[10:13] * 10
    # beam.c_p[15:17] = beam.c_p[15:17] * 10
    # beam.c_p[0:5] = beam.c_p[0:5] * 10
    p4 = runOptimizer(ba.UBeamAngleObjective, ric_flat, (beamlengths_p, cos0_t))
    plotResults(p1, p2, p3, p4)

