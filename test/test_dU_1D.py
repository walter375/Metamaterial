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
Tests the derivative function dU for the 1D rigidbody case
"""

import numpy as np
from code import beam_1D as rb


def test_dU_1D():
    positions_initial_ic = np.array([0.,1.,2.,3.])
    positions_pulled_ic = np.array([0.,1.,2.,3.])
    nb_hinges = len(positions_initial_ic)
    nb_bodies = len(positions_initial_ic)-1
    epsilon = 0.01

    k_n = np.ones(nb_bodies)
    beam_lengths_n = rb.getBeamLength_1D(positions_initial_ic) #shape(nb_hinges -1,)

    du_pulled = np.zeros([nb_hinges])
    du_epsilon = np.zeros([nb_hinges])

    u_pulled = np.zeros([nb_hinges])
    u_epsilon = np.zeros([nb_hinges])
    temp = np.zeros([nb_hinges])
    for i in range(len(positions_pulled_ic)):
        u_pulled[i] = rb.U_1D(positions_pulled_ic, beam_lengths_n, k_n)
        temp = rb.dU_1D(positions_pulled_ic, beam_lengths_n, k_n)
        du_pulled[i] = temp[i]

        positions_pulled_ic[i] = positions_pulled_ic[i]+epsilon

        u_epsilon[i] = rb.U_1D(positions_pulled_ic, beam_lengths_n, k_n)
        temp = rb.dU_1D(positions_pulled_ic, beam_lengths_n, k_n)
        du_epsilon[i] = temp[i]
        positions_pulled_ic = positions_initial_ic

    # print("\nu_pulled: ", u_pulled, "\nu_epsilon: ", u_epsilon)
    # print("\ndu_pulled:\n", du_pulled, "\ndu_epsilon:\n", du_epsilon)

    test_numerical = (u_epsilon - u_pulled)/epsilon
    test_analytical = (du_epsilon + du_pulled)/2
    # print("test_numerical: ", test_numerical, "test_analytical: ",test_analytical)
    np.testing.assert_allclose(test_numerical, test_analytical, rtol=0.001)