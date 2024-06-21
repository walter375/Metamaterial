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

global border
global r_stressed_ic

@pytest.mark.parametrize('positions_ic, i_p, j_p, optimizePos1, dim, optimizePos2', [
    ([[0, 0], [0, 1], [0, 2]], [0, 1], [1, 2], 1, 0, None),
    ([[0, 0], [1, 0], [2, 0]], [0, 1], [1, 2], 1, 1, None),
    ([[0, 0], [1, 1], [2, 2]], [0, 1], [1, 2], 1, 0, None),
])


def test_sensitivity( positions_ic, i_p, j_p, optimizePos1, dim, optimizePos2, epsilon=0.001):
    positions_ic = np.array(positions_ic, dtype=float)

    r_stressed_ic = np.zeros_like(positions_ic)
    r_stressed_ic += positions_ic
    r_stressed_ic[-1] += 0.5
    # global border
    border = rb.getBorderPoints(r_stressed_ic, left=1, right=1)
    nb_hinges, nb_dims = positions_ic.shape
    nb_bodies = len(i_p)
    i_p = np.array(i_p)
    j_p = np.array(j_p)

    c_p = np.ones(len(i_p))

    beam = rb.Beam(c_p, i_p, j_p, positions_ic, r_stressed_ic, border, x=1, y=1)
    beam_lengths_p = rb.getBeamLength(positions_ic, i_p, j_p)
    positions_flat = rb.ricFlat(r_stressed_ic, border, 1, 1)

    sensitivityAnalytical_2i = beam.displacementSensitivityObjective(c_p,
                                                            positions_flat,
                                                            beam_lengths_p,
                                                            positions_ic,
                                                            optimizePos1,
                                                            dim,
                                                            optimizePos2,
                                                            )
    sensitivityNumerical_2i = np.zeros_like(sensitivityAnalytical_2i, dtype=float)
    print("s",sensitivityAnalytical_2i)
    forces = np.zeros((2,2))
    for i in range(nb_bodies):
        diff_ic = np.zeros_like(c_p)
        diff_ic[i] = epsilon / 2

        c_minus_epsilon_half_ic = c_p - diff_ic
        c_plus_epsilon_half_ic = c_p + diff_ic

        displacement_minus_epsilon_half = beam.displacementObjective(c_minus_epsilon_half_ic,
                                                                     positions_flat,
                                                                     beam_lengths_p,
                                                                     positions_ic,
                                                                     optimizePos1,
                                                                     dim,
                                                                     optimizePos2
                                                                     )
        displacement_plus_epsilon_half = beam.displacementObjective(c_plus_epsilon_half_ic,
                                                                    positions_flat,
                                                                    beam_lengths_p,
                                                                    positions_ic,
                                                                    optimizePos1,
                                                                    dim,
                                                                    optimizePos2
                                                                    )
        displacements = ((displacement_plus_epsilon_half - displacement_minus_epsilon_half) / epsilon)
        print("d",displacements)
        #sensitivityNumerical_2i[i] = displacements
        print("sN",sensitivityNumerical_2i)
            # print(i,d, forces)
    #np.set_printoptions(formatter={'float': lambda x: "{0: 0.1f}".format(x)})
    #print("hessian numerical:\n", hessian_numerical)


    np.testing.assert_allclose(sensitivityAnalytical_2i, sensitivityNumerical_2i, rtol=1e-6, atol=1e-6)

