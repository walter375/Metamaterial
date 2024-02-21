#TODO: put all positions in positions, loop over positions
#TODO: write a seperate U_objective function for the optimizer
#TODO: rename variables
#TODO: tests


import numpy as np
import scipy

# Number of hinges and (elastic) bodies
nb_hinges = 4
nb_bodies = 3

# Positions of hinges (suffix _i for hinge and _c for Cartesian direction)
positions_ic = np.zeros((nb_hinges, 2))
# spring constant for every body (suffix _n for body)
k_n = np.zeros(nb_bodies)



def U(positions_ic, beam_lengths_n, k_n):
    U = 0.0
    for i in range(len(beam_lengths_n)):
        # first beam
        if i == 0:
            #print(i)
            U += 0.5 * k_n[i] * pow((positions_ic[i+1] - positions_ic[i] - beam_lengths_n[i]),2)
        # last beam
        elif i == (len(positions_ic)-1):
            #print("in elif U", i)
            U += 0.5 * k_n[i] * pow((positions_ic[i] - positions_ic[i-1] - beam_lengths_n[i]),2)
        # beams inbetween
        else:
            #print(i)
            U += 0.5 * k_n[i] * pow((positions_ic[i+1] - positions_ic[i] - beam_lengths_n[i]),2)
    return U

def dU(positions_ic, beam_lengths_n, k_n):
    dU = np.zeros(len(positions_ic))
    for i in range(len(positions_ic)):
        # first beam
        if i == 0:
            dU[i] = -k_n[i]*(positions_ic[i+1] - positions_ic[i] - beam_lengths_n[i])
        # last beam
        elif i == (len(positions_ic)-1):
            dU[i] = k_n[i-1]*(positions_ic[i] - positions_ic[i-1] - beam_lengths_n[i-1])
        # inbetween beams
        else:
            print(i)
            dU[i] = k_n[i-1]*(positions_ic[i] - positions_ic[i-1] - beam_lengths_n[i-1]) \
                    - k_n[i]*(positions_ic[i+1] - positions_ic[i] - beam_lengths_n[i])
    return dU

def U_objective(args):

    # U_objective(positions_opt_flat, test.getBeamLength(), c_alpha, positions_ic[0], positions_ic[-1])
    positions_ic, beamlengths_n, k_n = args[0], args[1], args[2]
    return U(positions_ic, beamlengths_n, k_n)

    # for i in range(len(beam_lengths_n)):
    #     U_obj = 0
    #     # first beam
    #     if i == 0:
    #         print("case (i==0): ", i)
    #         U_obj += 0.5 * c_alpha[i] * pow((positions_opt_flat[i] - boundary_1 - beam_lengths_n[i]), 2)
    #     # last beam
    #     elif i == (len(positions_opt_flat) - 1):
    #         print("case (i==len(positions_ic): ", i)
    #         U_obj += 0.5 * c_alpha[i] * pow((boundary_2 - positions_opt_flat[i-1]  - beam_lengths_n[i]), 2)
    #     # beams inbetween
    #     else:
    #         print("case else: ", i)
    #         U_obj += 0.5 * c_alpha[i] * pow((positions_opt_flat[i] - positions_opt_flat[i-1] - beam_lengths_n[i]), 2)
    # return U_obj


def main(self):
    positions = np.array((0,1,2,4))
    #pulledPositions = np.array((1,2,4))
    k = np.ones(len(positions)-1)  # internal nodes plus the two outer nodes (left and right)

    print("Pos: ", positions)
    print("k: ", k)
    print("beam lengths: ", self.getBeamLength(positions))
    print("U inital: ", self.U(positions, self.getBeamLength(positions), k))
    print("dU inital: ", self.dU(positions, self.getBeamLength(positions), k))
    print("U pulled: ", self.U(positions, self.getBeamLength(positions), k))
    print("dU pulled: ", self.dU(positions, self.getBeamLength(positions), k))

    #scipy.optimize.minimize(self.U(positions, self.getBeamLength(positions, 0, 7), k), (1,2,4))


class Structure:
    # Properties of bodies
    def __init__(self, positions_ic):
        self.positions_ic = positions_ic
    def getBeamLength_1D(self):
        beam_lengths_n = np.diff(self.positions_ic)

    def getBeamLength_2D(self):
        # beamlengths of bodies (_n for bodies)
        # calculate beam_length: sqrt((x_n+1 - x_n)² + (y_n+1 - y_n)²)
        # 1: np.diff(positions_ic, axis=0), 2.np.linalg.norm(position_diff_ic, axis=0)
        beam_lengths_n = np.linalg.norm(np.diff(self.positions_ic, axis=0), axis=1)
        return beam_lengths_n


if __name__ == "__main__":
    #test = Structure()
    #test.main()
    positions_ic = np.array([[0, 0],[1,2],[2,3],[3,5],[4,4]])
    k_n = np.ones(4)
    # positions_ic_diff = np.diff(positions_opt_flat, axis=0)
    # print("diff: ",positions_ic_diff)
    # print("norm: ",np.linalg.norm(positions_ic_diff, axis=1))

    test = Structure(positions_ic)
    # print("beamlength: ",test.getBeamLength())

    positions_opt_flat = positions_ic.reshape((10))
    print("postion flattened: ", positions_opt_flat, positions_opt_flat.shape)
#     print(positions_ic[0], positions_ic[-1])

    # U_objective([positions_opt_flat, test.getBeamLength(), c_alpha])
    # opt = scipy.optimize.minimize(U_objective(), x0=np.zeros([6]), args=(positions_opt_flat, test.getBeamLength(), c_alpha))
    # print(opt.message)