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
            
"""

#todo fix dU
#todo test!


def U_2D(positions_ic, beam_lengths_n, k_n, i_n, j_n):
    U = 0.0
    for m in range(len(i_n)):
        index_i = i_n[m]
        index_j = j_n[m]
        # print(positions_ic[index_j], positions_ic[index_i])
        U += 0.5 * k_n[m] * ((np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m])**2)
    return U

# dU should have for every position an x and y derivative -> dU.shape(positions, 2)
def dU_2D(positions_ic, beam_lengths_n, k_n, i_n, j_n):
    dU = np.zeros([len(positions_ic),2])
    r_hat = getRHat_2D(positions_ic, i_n, j_n)
    # loop over all beams
    for m in range(len(positions_ic)):
        # find all beams k connected to hinge m
        for k in range(len(k_n)):
            if (i_n[k] == m):
                index_j = j_n[k]
                index_i = i_n[k]
                # print("if: \n", m, k, positions_ic[index_i], positions_ic[index_j],\
                #       np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]), beam_lengths_n[k], r_hat[k])
                dU[m] += k_n[k] * (np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[k]) * r_hat[k]
            elif(j_n[k] == m):
                index_j = j_n[k]
                index_i = i_n[k]
#                 print("elif:\n", m, k, positions_ic[index_i], positions_ic[index_j], \
    #                 np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]), beam_lengths_n[k], r_hat[k])
                dU[m] += k_n[k] * (np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[k]) * r_hat[k]
            else:
                continue
    return dU

def getRHat_2D(positions_ic, i_n, j_n):
    r_hat = np.zeros([len(i_n), 2])
    for m in range(len(i_n)):
        index_i = i_n[m]
        index_j = j_n[m]
        r_hat[m] = ((positions_ic[index_j] - positions_ic[index_i]) / np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]))
    return r_hat

def getBeamLength_2D(positions_ic, i_n, j_n):
    # beamlengths of bodies (_n for bodies)
    # calculate beam_length: sqrt((x_n+1 - x_n)² + (y_n+1 - y_n)²)
    # 1: np.diff(positions_ic, axis=0), 2.np.linalg.norm(position_diff_ic, axis=0)
    beam_lengths_n = np.zeros(len(i_n))
    for m in range(len(i_n)):
        index_i = i_n[m]
        index_j = j_n[m]
        beam_lengths_n[m] = np.linalg.norm(positions_ic[index_j] - positions_ic[index_i])
    return beam_lengths_n

def objective(positions_flat_2i, beam_lengths_n, k_n, i_n, j_n):
    positions_ic = positions_flat_2i.reshape(nb_hinges, 2)
    #print(positions_ic)
    return U_2D(positions_ic, beam_lengths_n, k_n, i_n, j_n)

# first position needs to stay zero
def con1(positions_flat_2i):
    # print("con1: ", positions_2i[0:2], start)
    return start - positions_flat_2i[0:2]

# last position needs to stay the same, cannot move
def con2(positions_flat_2i):
    # print("\ncon2: ",positions_2i[-2:], end)
    return positions_flat_2i[-2:] - end

if __name__ == "__main__":
    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [6, 1]])
    i_n = np.array([0, 1, 2, 1, 4, 3])
    j_n = np.array([1, 2, 3, 4, 3, 5])

    nb_bodies = len(i_n)
    nb_hinges = len(positions_initial_ic)
    positions_flat = positions_final_ic.reshape(nb_hinges * 2)
    k_n = np.ones(nb_bodies)
    start = positions_initial_ic[0]
    end = positions_final_ic[-1]

    beam_lengths_n = getBeamLength_2D(positions_initial_ic, i_n, j_n)
    print("obj: ", objective(positions_flat, beam_lengths_n, k_n, i_n, j_n))
    print("dU: ", dU_2D(positions_final_ic, beam_lengths_n, k_n, i_n, j_n))
    cons = [{'type': 'eq', 'fun': con1},
            {'type': 'eq', 'fun': con2}]
    res = scipy.optimize.minimize(objective, x0=np.zeros(nb_hinges*2), args=(beam_lengths_n, k_n, i_n, j_n), constraints=cons)#,jac=dU_2D)
    print("msg: ", res.message)
    print("res.x:\n ", res.x.reshape(nb_hinges, 2))
    print("fun:\n ", res.fun)

    points= res.x.reshape(nb_hinges,2)
    print(points[:,0], points[:,1])
    plt.plot(positions_initial_ic[:,0], positions_initial_ic[:,1], 'o')
    plt.plot(points[:,0], points[:,1], 'x')
    plt.show()