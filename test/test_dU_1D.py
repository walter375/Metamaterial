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

    dx = 0.001
    nb_points = 1000
    x = np.arange(nb_points) * dx

    y = rb.U_1D(x, np.ones(nb_points-1), np.ones(nb_points-1))

    test = ((x[:-1] + x[1:]) / 2)

    dy_analytical = rb.dU_1D(test, np.ones(nb_points-1), np.ones(nb_points-1))
    dy_numerical = np.diff(y) / np.diff(x)

    #np.testing.assert_allclose(dy_analytical, dy_numerical)

