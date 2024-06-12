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
import numpy as np
import pytest
from Code import beam_2D as rb
@pytest.mark.parametrize('positions_ic, i_p, j_p', [
    ([[0, 0], [0, 1], [0, 2]], [0, 1], [1, 2]),
    ([[0, 0], [1, 0], [2, 0]], [0, 1], [1, 2]),
    ([[0, 0], [1, 1], [2, 2]], [0, 1], [1, 2]),
    ([[0, 0], [1, 1], [2, 2], [1, 3]], [0, 1, 0, 2], [1, 2, 3, 3]),
    ([[0, 1], [1, 1], [2, 2], [3, 1], [2, 0], [4, 1]], [0, 1, 2, 1, 4, 3], [1, 2, 3, 4, 3, 5])
])

def test_hessian(positions_ic, i_p, j_p, epsilon=0.00001):
    positions_ic = np.array(positions_ic, dtype=float)
    nb_hinges, nb_dims = positions_ic.shape
    # nb_bodies = len(i_p)
    i_p = np.array(i_p)
    j_p = np.array(j_p)
    c_p = np.ones(len(i_p))

    beam = rb.Beam(c_p, i_p, j_p)
    beam_lengths_p = rb.getBeamLength(positions_ic, i_p, j_p)

    hessian_2i2i = beam.getHessianBeam(positions_ic, beam_lengths_p)
    hessian_numerical = np.zeros_like(hessian_2i2i, dtype=float)
    #print("hessian analytical:\n", hessian_2i2i)
    forces = np.zeros((2,2))
    for i in range(nb_hinges):
        for d in range(nb_dims):
            diff_ic = np.zeros_like(positions_ic)
            diff_ic[i, d] = epsilon / 2

            positions_minus_epsilon_half_ic = positions_ic - diff_ic
            positions_plus_epsilon_half_ic = positions_ic + diff_ic

            gradient_minus_epsilon_half = beam.getGradientBeam(positions_minus_epsilon_half_ic, beam_lengths_p)
            gradient_plus_epsilon_half = beam.getGradientBeam(positions_plus_epsilon_half_ic, beam_lengths_p)

            forces = ((gradient_plus_epsilon_half - gradient_minus_epsilon_half) / epsilon).flatten()
            hessian_numerical[:, i*2+d] = forces
            # print(i,d, forces)
    #np.set_printoptions(formatter={'float': lambda x: "{0: 0.1f}".format(x)})
    #print("hessian numerical:\n", hessian_numerical)


    np.testing.assert_allclose(hessian_2i2i, hessian_numerical, rtol=1e-6, atol=1e-6)

