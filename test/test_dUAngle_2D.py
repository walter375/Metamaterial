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
"""
import pytest

import numpy as np
from code import beam_2D as rb


@pytest.mark.parametrize('positions_ic,i_t,j_t,k_t', [
    ([[0, 0], [1, 0], [1, 1]], [0], [1], [2]),
    ([[0.3, 0.1], [1, -0.1], [1.2, 1]], [0], [1], [2]),
    ([[0.3, 0.1], [1, -0.1], [1.2, 1]], [2], [0], [1]),
    ([[0.3, 0.1], [1, -0.1], [1.2, 1]], [0, 2], [1, 0], [2, 1]),
])
def test_dU_2D_xy(positions_ic, i_t, j_t, k_t, epsilon=0.001):
    positions_ic = np.array(positions_ic, dtype=float)  # shape=(nb_hinges, 2)
    nb_hinges, nb_dims = positions_ic.shape
    i_t = np.array(i_t)
    j_t = np.array(j_t)
    k_t = np.array(k_t)

    c_t = np.ones(len(i_t))
    # cos0_t = rb.getCosAngles_2D(positions_ic, i_t, j_t, k_t)
    cos0_t = np.ones_like(c_t)

    cos_t = rb.getCosAngles_2D(positions_ic, i_t, j_t, k_t)

    # u = rb.UAngle_2D(cos_t, cos0_t, c_t)
    du_ic = rb.dUAngle_2D(positions_ic, cos_t, cos0_t, c_t, i_t, j_t, k_t)

    for i in range(nb_hinges):
        for d in range(nb_dims):
            diff_ic = np.zeros_like(positions_ic)
            diff_ic[i, d] = epsilon / 2

            positions_minus_epsilon_half_ic = positions_ic - diff_ic
            positions_plus_epsilon_half_ic = positions_ic + diff_ic

            cosMinusEpsilonHalf_t = rb.getCosAngles_2D(positions_minus_epsilon_half_ic, i_t, j_t, k_t)
            cosPlusEpsilonHalf_t = rb.getCosAngles_2D(positions_plus_epsilon_half_ic, i_t, j_t, k_t)

            u_minus_epsilon_half = rb.UAngle_2D(cosMinusEpsilonHalf_t, cos0_t, c_t)
            u_plus_epsilon_half = rb.UAngle_2D(cosPlusEpsilonHalf_t, cos0_t, c_t)

            du_numerical = (u_plus_epsilon_half - u_minus_epsilon_half) / epsilon

            #print("analytic, numerical: ", i, d, du_ic[i, d], du_numerical)
            np.testing.assert_allclose(du_ic[i, d], du_numerical, rtol=1e-5)
