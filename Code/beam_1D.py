import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""
0 - 1 - 2 - 3
0 - 1 - 2 ------- 7
"""


def U_1D(positions_ic, beam_lengths_n, k_n):
    U = 0.0
    for i in range(len(k_n)):
        # # last beam
        if i == (len(k_n)):
            print(i, k_n.shape, beam_lengths_n.shape)
            U += 0.5 * k_n[i] * pow((positions_ic[i] - positions_ic[i - 1] - beam_lengths_n[i]), 2)
        # first beam and inbetween beams
        else:
            U += 0.5 * k_n[i] * pow((positions_ic[i + 1] - positions_ic[i] - beam_lengths_n[i]), 2)
    return U

def dU_1D(positions_ic, beam_lengths_n, k_n):
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
            dU[i] = k_n[i-1]*(positions_ic[i] - positions_ic[i-1] - beam_lengths_n[i-1]) \
                    - k_n[i]*(positions_ic[i+1] - positions_ic[i] - beam_lengths_n[i])
    return dU

def ddU_1D()
def getBeamLength_1D(positions_i):
    beam_lengths_n = np.diff(positions_i)
    return beam_lengths_n

def objective(positions_i, beam_lengths_n, k_n):
    return U_1D(positions_i, beam_lengths_n,k_n)

# first position needs to stay zero
def con1(positions_ic):
    # start = 0
    return start - positions_ic[0]

# last position needs to stay the same, cannot move
def con2(positions_ic):
    # end = 8
    return positions_ic[-1] - end

if __name__ == "__main__":
    positions_initial_i = np.array([0, 1, 2, 3, 4, 8])
    positions_final_i = np.array([0, 1, 2, 3, 4, 8])
    nb_hinges = len(positions_initial_i)
    nb_bodies = nb_hinges - 1

    k_n = np.ones(nb_bodies)

    # for constraints
    start = positions_initial_i[0]
    end = positions_final_i[-1]

    #positions_ic_flat = positions_ic.reshape(nb_positions*2)
    beam_lengths_n = getBeamLength_1D(positions_initial_i)
    print("u1: ", U_1D(positions_initial_i, beam_lengths_n, k_n))
    print("u2: ", U_1D(positions_final_i, beam_lengths_n, k_n))
    print("du1: ", dU_1D(positions_initial_i, beam_lengths_n, k_n))
    print("du2: ", dU_1D(positions_final_i, beam_lengths_n, k_n))
    cons = ({'type': 'eq', 'fun': con1},
            {'type': 'eq', 'fun': con2})
    res = scipy.optimize.minimize(objective, x0=np.zeros(len(positions_initial_i)),args=(beam_lengths_n, k_n), constraints=cons, jac=dU_1D)
    print(res.message)
    print(res.x)
    print(res.fun)

    plt.plot(res.x, np.zeros(nb_hinges), '-o')
    plt.show()



