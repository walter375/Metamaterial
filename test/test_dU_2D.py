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
Tests the derivative function dU for the 2D rigidbody case
"""

import numpy as np
import math
from code import rigidbody_2D as rb
def test_dU_2D():
    epsilon = 0.001
    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_pulled_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4,0]])
    positions_epsilon_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4,0+epsilon]])
    positions_epsilon_half_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4,0+epsilon/2]])
    i_n = np.array([0, 1, 2, 1, 4, 3])
    j_n = np.array([1, 2, 3, 4, 3, 5])

    # positions_initial_ic = np.array([[0.,0.], [0.,1.], [0.,2.],[0.,3.]])
    # positions_pulled_ic = np.array([[0.,0.], [0.,1.], [0.,2.],[1.,3.]])
    # positions_epsilon_ic = np.array([[0.,0.],[0.,1.], [0.,2.],[1+ epsilon,3.]])
    # positions_epsilon_half_ic = np.array([[0., 0.], [0., 1.], [0., 2.], [1. + epsilon/2 , 3. ]])
    # i_n = np.array([0,1,2])
    # j_n = np.array([1,2,3])

    nb_bodies = len(i_n)
    nb_hinges = len(positions_initial_ic)
    k_n = np.ones(nb_bodies)

    beam_lengths_n = rb.getBeamLength_2D(positions_initial_ic, i_n, j_n)

    u_initial = rb.U_2D(positions_initial_ic, beam_lengths_n, k_n, i_n, j_n)
    u_pulled = rb.U_2D(positions_pulled_ic, beam_lengths_n, k_n, i_n, j_n)
    u_epsilon = rb.U_2D(positions_epsilon_ic, beam_lengths_n, k_n, i_n, j_n)

    du_initial = rb.dU_2D(positions_initial_ic, beam_lengths_n, k_n, i_n, j_n)
    du_pulled = rb.dU_2D(positions_pulled_ic, beam_lengths_n, k_n, i_n, j_n)
    du_epsilon = rb.dU_2D(positions_epsilon_ic, beam_lengths_n, k_n, i_n, j_n)
    du_epsilon_half = rb.dU_2D(positions_epsilon_half_ic, beam_lengths_n, k_n, i_n, j_n)

    print("\nu_inital: ", u_initial, "\nu_pulled: ", u_pulled, "\nu_epsilon: ", u_epsilon, "\ndu_initial: ", du_initial, "\ndu_pulled: ", du_pulled, "\ndu_epsilon: ", du_epsilon, "\ndu_epsilon2: ", du_epsilon_half)

    test_numerical = (u_epsilon-u_pulled)/epsilon

    # print(np.linalg.norm(positions_final_ic[-1]-positions_initial_ic[-1]))
    print("\ntest_num: ", test_numerical)
    #diff = (du_pulled[-1] - du_inital[-1])
    #print(diff)
    test_analytical = du_epsilon_half[-1:,1]
    print("\nanalytical: ", test_analytical)

    # print("dy_analytical:", dy_analytical.shape)
    # print("dy_numerical:", dy_numerical.shape)
    np.testing.assert_allclose(test_analytical, test_numerical, rtol=1e-6, atol=1e-5)