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
import math
from code import beam_2D as rb

epsilon = 0.1
positions_initial_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.5]], dtype=float)  # shape=(nb_hinges, 2)
positions_pulled_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)

i_p = np.array([0, 0, 1, 3, 1])
j_p = np.array([2, 1, 3, 2, 2])

i_t = np.array([1,3])
j_t = np.array([0,2])
k_t = np.array([2,1])

nb_bodies = len(i_p)
nb_hinges = len(positions_initial_ic)
beam_lengths_n = rb.getBeamLength_2D(positions_initial_ic, i_p, j_p)

c_t3 = np.ones((len(i_t), 3))

beamlengths0ij_t = rb.getBeamLength_2D(positions_initial_ic, i_t, j_t)
beamlengths0kj_t = rb.getBeamLength_2D(positions_initial_ic, k_t, j_t)
beamlengths0ik_t = rb.getBeamLength_2D(positions_initial_ic, i_t, k_t)

u_pulled = rb.UTriplet_2D(positions_pulled_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
du = rb.dUTriplet_2D(positions_pulled_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
def test_dU_2D_xy():
    positions_epsilon_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)
    positions_epsilon_half_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)

    u_epsilon = np.zeros(nb_hinges, dtype=float)
    du_epsilon_half = np.zeros([nb_hinges,2], dtype=float)
    for i in range(nb_hinges):
        positions_epsilon_ic[i] = positions_epsilon_ic[i] + epsilon
        positions_epsilon_half_ic[i] = positions_epsilon_half_ic[i] + epsilon/2
        u_epsilon[i] = rb.UTriplet_2D(positions_epsilon_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
        temp = rb.dUTriplet_2D(positions_epsilon_half_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
        du_epsilon_half[i] = temp[i]
        positions_epsilon_ic[i] = positions_pulled_ic[i]
        positions_epsilon_half_ic[i] = positions_pulled_ic[i]

    test_numerical = (u_epsilon-u_pulled)/epsilon
    test_analytical = np.sum(du_epsilon_half, axis=1)
    # print("\nxy\ntest_analytical:\n", test_analytical,"\n", "\ntest_numerical:\n", test_numerical)
    np.testing.assert_allclose(test_analytical, test_numerical, rtol=1e-3, atol=1e-3)

def test_dU_2D_x():
    epsilon = 0.1
    positions_epsilon_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)
    positions_epsilon_half_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)

    u_epsilon = np.zeros(nb_hinges, dtype=float)
    du_epsilon_half = np.zeros([nb_hinges,2], dtype=float)
    for i in range(nb_hinges):
        positions_epsilon_ic[i,0] = positions_epsilon_ic[i,0] + epsilon
        positions_epsilon_half_ic[i,0] = positions_epsilon_half_ic[i,0] + epsilon/2
        u_epsilon[i] = rb.UTriplet_2D(positions_epsilon_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
        temp = rb.dUTriplet_2D(positions_epsilon_half_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3, i_t, j_t, k_t)
        du_epsilon_half[i] = temp[i]
        positions_epsilon_ic[i] = positions_pulled_ic[i]
        positions_epsilon_half_ic[i] = positions_pulled_ic[i]

    test_numerical = (u_epsilon-u_pulled)/epsilon
    test_analytical = du_epsilon_half[:, 0]
    # print("\nx\ntest_analytical:\n", test_analytical,"\n", "\ntest_numerical:\n", test_numerical)
    np.testing.assert_allclose(test_analytical, test_numerical, rtol=1e-3, atol=1e-3)


def test_dU_2D_y():
    epsilon = 0.1
    positions_epsilon_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)
    positions_epsilon_half_ic = np.array([[0, 0], [1.2, 1], [2,0], [3, 1.7]], dtype=float)

    u_epsilon = np.zeros(nb_hinges, dtype=float)
    du_epsilon_half = np.zeros([nb_hinges, 2], dtype=float)
    for i in range(nb_hinges):
        positions_epsilon_ic[i,1] = positions_epsilon_ic[i,1] + epsilon
        positions_epsilon_half_ic[i,1] = positions_epsilon_half_ic[i,1] + epsilon / 2
        u_epsilon[i] = rb.UTriplet_2D(positions_epsilon_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3,
                                      i_t, j_t, k_t)
        temp = rb.dUTriplet_2D(positions_epsilon_half_ic, beamlengths0ij_t, beamlengths0kj_t, beamlengths0ik_t, c_t3,
                               i_t, j_t, k_t)
        du_epsilon_half[i] = temp[i]
        positions_epsilon_ic[i] = positions_pulled_ic[i]
        positions_epsilon_half_ic[i] = positions_pulled_ic[i]
    test_numerical = (u_epsilon-u_pulled)/epsilon
    test_analytical = du_epsilon_half[:, 1]
    # print("\ny\ntest_analytical:\n", test_analytical,"\n", "\ntest_numerical:\n", test_numerical)
    np.testing.assert_allclose(test_analytical, test_numerical, rtol=1e-3, atol=1e-3)