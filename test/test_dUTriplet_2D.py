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


"""
Tests the derivative function dUTriplet_2D for the 2D rigidbody case
"""

import numpy as np
import pytest
from Code import beam_2D as rb


@pytest.mark.parametrize('positions_ic,i_t,j_t,k_t', [
    ([[0, 0], [1, 0], [1, 1]], [0], [1], [2]),
    ([[0.2, 0.1], [1, -0.5], [1.3, 0.9]], [0], [1], [2]),
    ([[0.2, 0.1], [1, -0.5], [1.3, 0.9]], [2], [0], [1]),
    ([[0.2, 0.1], [1, -0.5], [1.3, 0.9]], [2,1], [0,2], [1,0]),
    ([[0.2, 0.1], [1, -0.5], [1.3, 0.9], [3, 1.5]], [2,1], [0,2], [1,3])
])
def test_dU(positions_ic, i_t, j_t, k_t, epsilon=0.001):
    positions_ic = np.array(positions_ic, dtype=float)
    nb_hinges, nb_dims = positions_ic.shape
    i_t = np.array(i_t)
    j_t = np.array(j_t)
    k_t = np.array(k_t)
    c_t3 = np.ones((len(i_t), 3))

    triplet = rb.Triplet(c_t3, i_t, j_t, k_t)

    beamlengths0ij_t = rb.getBeamLength(positions_ic, i_t, j_t)
    beamlengths0kj_t = rb.getBeamLength(positions_ic, k_t, j_t)
    beamlengths0ik_t = rb.getBeamLength(positions_ic, i_t, k_t)

    du_ic = triplet.dUTriplet(positions_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t)

    for i in range(nb_hinges):
        for d in range(nb_dims):
            diff_ic = np.zeros_like(positions_ic)
            diff_ic[i, d] = epsilon / 2

            positions_minus_epsilon_half_ic = positions_ic - diff_ic
            positions_plus_epsilon_half_ic = positions_ic + diff_ic

            u_minus_epsilon_half = triplet.UTriplet(positions_minus_epsilon_half_ic, beamlengths0ij_t, beamlengths0kj_t,
                                                  beamlengths0ik_t)
            u_plus_epsilon_half = triplet.UTriplet(positions_plus_epsilon_half_ic, beamlengths0ij_t, beamlengths0kj_t,
                                                 beamlengths0ik_t)
            du_numerical = (u_plus_epsilon_half - u_minus_epsilon_half) / epsilon

            # print("analytic, numerical: ", du_ic[i, d], du_numerical)
            np.testing.assert_allclose(du_ic[i, d], du_numerical, rtol=1e-6, atol=1e-6)
