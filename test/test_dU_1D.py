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
from code import rigidbody_1D as rb


def test_dU_1D():
    positions_initial_ic = np.array([0.,1.,2.,3.])
    positions_pulled_ic = np.array([0.,1.,2.,3.])
    nb_hinges = len(positions_initial_ic)
    nb_bodies = len(positions_initial_ic)-1
    epsilon = 0.01

    k_n = np.ones(nb_bodies)
    beam_lengths_n = rb.getBeamLength_1D(positions_initial_ic) #shape(nb_hinges -1,)

    u_pulled = rb.U_1D(positions_pulled_ic, beam_lengths_n, k_n)
    du_pulled = rb.dU_1D(positions_pulled_ic, beam_lengths_n, k_n)  # ((positions_pulled_ic[:-1]+positions_pulled_ic[1:])/2)

    positions_pulled_ic[-2] = positions_pulled_ic[-2]+epsilon

    u_epsilon = rb.U_1D(positions_pulled_ic, beam_lengths_n, k_n)
    du_epsilon = rb.dU_1D(positions_pulled_ic, beam_lengths_n, k_n)

    print("\nu_pulled: ", u_pulled, "\nu_epsilon: ", u_epsilon)
    print("\ndu_pulled: ", du_pulled, "\ndu_epsilon: ", du_epsilon)

    test_numerical = (u_epsilon - u_pulled)/epsilon
    print("test_numerical: ", test_numerical)
    test_analytical = (du_epsilon[-2] + du_pulled[-2])/2
    print("test_analytical: ",test_analytical)
    np.testing.assert_allclose(test_numerical, test_analytical)