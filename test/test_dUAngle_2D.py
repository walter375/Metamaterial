#
# Copyright 2021 Lars Pastewka
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Tests the derivative function dUAngle_2D for the 2D rigidbody case

   1 ___ 2
 /  \   /
0 ___ 3

"""
import math

import numpy as np
from code import beam_2D as rb

positions_initial_ic = np.array([[0, 0], [1, 0], [1.1,1]], dtype=float)  # shape=(nb_hinges, 2)
positions_pulled_ic = np.array([[0, 0], [1, 0], [1,1]], dtype=float)

# i_alpha = np.array([0, 0, 1, 3, 1])
# j_p = np.array([3, 1, 3, 2, 2])

#i_t = np.array([1,3,0,1,2,3])
#j_t = np.array([0,1,3,3,1,2])
#k_t = np.array([3,0,1,2,3,1])

i_alpha = np.array([0, 0])
j_alpha = np.array([1, 2])

i_beta = np.array([1])
j_beta = np.array([0])
k_beta = np.array([2])

nb_bodies = len(i_alpha)
nb_hinges = len(positions_initial_ic)
beam_lengths_n = rb.getBeamLength_2D(positions_initial_ic, i_alpha, j_alpha)

c_beta = np.ones(len(i_beta))

u_pulled = rb.UAngle_2D(positions_initial_ic, positions_pulled_ic, c_beta, i_beta, j_beta, k_beta)
#du_pulled = rb.dUBeam_2D(positions_pulled_ic, beamlengths_p, k_n, i_i, j_i)
def test_dU_2D_xy():

    positions_epsilon_ic = np.array([[0, 0], [1, 0], [0,1]], dtype=float)
    positions_epsilon_half_ic = np.array([[0, 0], [1, 0], [0,1]], dtype=float)
    u_epsilon = rb.UAngle_2D(positions_initial_ic, positions_epsilon_ic, c_beta, i_beta, j_beta, k_beta)
    du_epsilon_half = rb.dUAngle_2D(positions_initial_ic, positions_epsilon_half_ic, c_beta, i_beta, j_beta, k_beta)

    # u_epsilon = np.zeros(len(i_t))
    # du_epsilon_half = np.zeros(len(i_t))
    #
    # for i in range(len(i_t)):
    #     positions_epsilon_ic[i] = positions_epsilon_ic[i] + epsilon
    #     positions_epsilon_half_ic[i] = positions_epsilon_half_ic[i] + epsilon/2
    #     u_epsilon[i] = rb.UAngle_2D(positions_initial_ic, positions_epsilon_ic, c_t, i_t, j_t, k_t)
    #     temp = rb.dUAngle_2D(positions_initial_ic, positions_epsilon_half_ic, c_t, i_t, j_t, k_t)
    #     du_epsilon_half[i] = temp[i]
    #     positions_epsilon_ic[i] = positions_pulled_ic[i]
    #     positions_epsilon_half_ic[i] = positions_pulled_ic[i]

    print("\nu_epsilon: ", u_epsilon, "\nu_pulled: ", u_pulled)
    print("\ndu_epsilon_half: ", du_epsilon_half)
    test_numerical = (u_epsilon-u_pulled)/math.cos(math.radians(45))
    test_analytical = du_epsilon_half
    print("\ntest_analytical: ", test_analytical, "\ntest_numerical: ", test_numerical)
    # np.testing.assert_allclose(test_analytical, test_numerical, rtol=1e-6, atol=1e-5)

