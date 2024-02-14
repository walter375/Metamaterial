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
    positions_initial_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]])  # shape=(nb_hinges, 2)
    positions_final_ic = np.array([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1.0001]])
    i_n = np.array([0, 1, 2, 1, 4, 3])
    j_n = np.array([1, 2, 3, 4, 3, 5])

    nb_bodies = len(i_n)
    nb_hinges = len(positions_initial_ic)
    positions_flat = positions_final_ic.reshape(nb_hinges * 2)
    k_n = np.ones(nb_bodies)
    start = positions_initial_ic[0]
    end = positions_final_ic[-1]

    beam_lengths_n = rb.getBeamLength_2D(positions_initial_ic, i_n, j_n)

    u_inital = rb.U_2D(positions_initial_ic, beam_lengths_n, k_n, i_n, j_n)
    u_final = rb.U_2D(positions_final_ic, beam_lengths_n, k_n, i_n, j_n)

    du_inital = rb.dU_2D(positions_initial_ic, beam_lengths_n, k_n, i_n, j_n)
    print("\ndu_pulled\n")
    du_pulled = rb.dU_2D(positions_final_ic, beam_lengths_n, k_n, i_n, j_n)

    print("\nu_inital: ", u_inital, "\nu_final: ", u_final, "\ndu_inital: ", du_inital, "\ndu_pulled: ", du_pulled)
    epsilon = np.linalg.norm(positions_final_ic[-1]-positions_initial_ic[-1]) - np.linalg.norm(positions_initial_ic[-1]-positions_initial_ic[-1])
    test_numerical = (u_final-u_inital) / epsilon
    # print(np.linalg.norm(positions_final_ic[-1]-positions_initial_ic[-1]))
    print("\ntest_num: ", test_numerical)

    test_analytical = np.sum(du_inital[-1] + du_pulled[-1])    # print("\ntest_analytical: ", test_analytical)
    print("\nanalytical: ", test_analytical)

    # print("dy_analytical:", dy_analytical.shape)
    # print("dy_numerical:", dy_numerical.shape)
    #np.testing.assert_allclose(test_analytical, test_numerical, atol=1e-15)