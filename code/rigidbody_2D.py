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
nb_bodies = 6
nb_hinges = 6
positions_initial_ic = np.array([[0,1], [1,1], [2,2], [3,1], [2,0],[4,1]]) # shape=(nb_hinges, 2)
positions_final_ic = np.array([[0,1], [1,1], [2,2], [3,1], [2,0],[6,1]])
positions_flat = positions_final_ic.reshape(nb_hinges * 2)
start = positions_initial_ic[0]
end = positions_final_ic[-1]

i_n = np.array([0,1,2,1,4,3])
j_n = np.array([1,2,3,4,3,5])

k_n = np.ones(nb_bodies)

def U_2D(positions_ic, beam_lengths_n, k_n):
    U = 0.0
    #print("pos:\n", positions_ic)
    for m in range(nb_bodies):
        # first beam
        index_i = i_n[m]
        index_j = j_n[m]
        # print(positions_ic[index_j], positions_ic[index_i])
        U += 0.5 * k_n[m] * ((np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[m])**2)
    return U

def dU(positions_ic, beam_lengths_n, k_n):
    dU = np.zeros(nb_bodies)
    # loop over all beams
    for m in range(nb_bodies):
        # find all beams connected to hinge i
        for k in range(nb_bodies):
            if (i_n[k] == m or j_n[k] == m):
                index_j = j_n[k]
                index_i = i_n[k]
                dU[m] += k_n[k] * (np.linalg.norm(positions_ic[index_j] - positions_ic[index_i]) - beam_lengths_n[k])
            else:
                continue
    return dU

def getBeamLength_2D(positions_ic):
    # beamlengths of bodies (_n for bodies)
    # calculate beam_length: sqrt((x_n+1 - x_n)² + (y_n+1 - y_n)²)
    # 1: np.diff(positions_ic, axis=0), 2.np.linalg.norm(position_diff_ic, axis=0)
    beam_lengths_n = np.zeros(nb_bodies)
    for m in range(nb_bodies):
        index_i = i_n[m]
        index_j = j_n[m]
        beam_lengths_n[m] = np.linalg.norm(positions_ic[index_j] - positions_ic[index_i])
    return beam_lengths_n

def objective(positions_flat_2i, beam_lengths_n, k_n):
    positions_ic = positions_flat_2i.reshape(nb_hinges, 2)
    #print(positions_ic)
    return U_2D(positions_ic, beam_lengths_n, k_n)

# first position needs to stay zero
def con1(positions_flat_2i):
    # print("con1: ", positions_2i[0:2], start)
    return start - positions_flat_2i[0:2]

# last position needs to stay the same, cannot move
def con2(positions_flat_2i):
    # print("\ncon2: ",positions_2i[-2:], end)
    return positions_flat_2i[-2:] - end

if __name__ == "__main__":
    beam_lengths_n = getBeamLength_2D(positions_initial_ic)
    print("obj: ", objective(positions_flat, beam_lengths_n, k_n))
    print("dU: ", dU(positions_final_ic, beam_lengths_n, k_n))
    cons = [{'type': 'eq', 'fun': con1},
            {'type': 'eq', 'fun': con2}]
    res = scipy.optimize.minimize(objective, x0=np.zeros(nb_hinges*2), args=(beam_lengths_n, k_n), constraints=cons) #,jac=dU)
    print("msg: ", res.message)
    print("x: ", res.x.reshape(nb_hinges, 2))
    print("fun: ", res.fun)
    #
    # points= res.x.reshape(nb_hinges,2)
    # print(points[:,0], points[:,1])
    # plt.plot(positions_initial_ic[:,0], positions_initial_ic[:,1], 'o')
    # plt.plot(points[:,0], points[:,1], 'x')
    # plt.show()