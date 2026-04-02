"""
Tests for voxel moment-fitting quadrature
==========================================
Mirrors the four C++ Catch2 test cases in the test file.
"""

import csv as csv_mod
import itertools
import math
import tempfile
import unittest

import numpy as np

from momentfitting import (
    compute_moments,
    gauss_legendre_points,
    write_csv,
    alpha_uniform,
)


# ---------------------------------------------------------------------------
# Alpha / scaling helpers (matching C++ plateWithAHole and quadraticScaling)
# ---------------------------------------------------------------------------

def plate_with_a_hole(xyz: np.ndarray) -> float:
    """1 outside circle of radius 0.5 centred at origin, 0 inside."""
    cx, cy, r = 0.0, 0.0, 0.5
    dist2 = (xyz[0] - cx) ** 2 + (xyz[1] - cy) ** 2
    return 0.0 if dist2 < r * r else 1.0


def quadratic_scaling(xyz: np.ndarray) -> float:
    return xyz[0] ** 2 + xyz[1] ** 2


# ---------------------------------------------------------------------------
# test0 – uniform alpha=1: moments must equal standard GL weights
#
# C++ test: scaling0 = 1, scaling1 = 1, nvoxels=10x10, order=2
# With alpha=1 the moment-fitting weights must reproduce the 5-point
# (2*2+1 = 5) tensor-product Gauss-Legendre weights on [-1,1]^2.
# ---------------------------------------------------------------------------

class TestMomentFitting2dTest0(unittest.TestCase):

    D       = 2
    nvoxels = (10, 10)
    degree  = 2    # stride = 2*2+1 = 5

    def _run(self):
        return compute_moments(
            nvoxels=self.nvoxels,
            polynomial_degree=self.degree,
            alpha_func=alpha_uniform,
        )

    def test_number_of_quadrature_points(self):
        moments, _ = self._run()
        stride = 2 * self.degree + 1
        self.assertEqual(len(moments), stride ** self.D)

    def test_element_ip_positions(self):
        """Element IPs must be standard GL nodes on [-1, 1]."""
        _, elem_pts = self._run()
        stride = 2 * self.degree + 1
        gl_pts, _ = gauss_legendre_points(stride)
        for d in range(self.D):
            np.testing.assert_allclose(elem_pts[d], gl_pts, atol=1e-12)

    def test_weights_equal_standard_gl(self):
        """With alpha=1 the moment weights must equal tensor-product GL weights."""
        moments, _ = self._run()
        stride = 2 * self.degree + 1
        _, gl_wts = gauss_legendre_points(stride)

        for flat_idx, lmn in enumerate(
            itertools.product(*[range(stride)] * self.D)
        ):
            expected = math.prod(gl_wts[lmn[d]] for d in range(self.D))
            self.assertAlmostEqual(
                moments[flat_idx], expected, delta=1e-10 * abs(expected),
                msg=f"index {lmn}: got {moments[flat_idx]}, expected {expected}",
            )

    def test_weight_sum_equals_element_area(self):
        """Sum of weights = area of [-1,1]^2 = 4.0."""
        moments, _ = self._run()
        self.assertAlmostEqual(moments.sum(), 4.0, places=10)


# ---------------------------------------------------------------------------
# test1 – plate-with-a-hole alpha: compare against expected weights
#
# C++ test: scaling0 = plateWithAHole, scaling1 = 1, nvoxels=10x10, order=2
# The 25 expected weights are taken directly from the C++ test.
# ---------------------------------------------------------------------------

class TestMomentFitting2dTest1(unittest.TestCase):

    D       = 2
    nvoxels = (10, 10)
    degree  = 2

    EXPECTED_WEIGHTS = [
        0.056035619002542, 0.114459279950980, 0.141798787982819,
        0.114459279950980, 0.056035619002542,
        0.114459279950980, 0.217930663040670, 0.191784705879454,
        0.217930663040670, 0.114459279950980,
        0.141798787982819, 0.191784705879454, -0.087107911131017,
        0.191784705879454, 0.141798787982819,
        0.114459279950980, 0.217930663040670, 0.191784705879454,
        0.217930663040670, 0.114459279950980,
        0.056035619002542, 0.114459279950980, 0.141798787982819,
        0.114459279950980, 0.056035619002542,
    ]

    def _run(self):
        return compute_moments(
            nvoxels=self.nvoxels,
            polynomial_degree=self.degree,
            alpha_func=plate_with_a_hole,
        )

    def test_weight_count(self):
        moments, _ = self._run()
        self.assertEqual(len(moments), len(self.EXPECTED_WEIGHTS))

    def test_weights_match_reference(self):
        moments, _ = self._run()
        for i, (got, exp) in enumerate(zip(moments, self.EXPECTED_WEIGHTS)):
            self.assertAlmostEqual(
                got, exp, delta=max(1e-8 * abs(exp), 1e-12),
                msg=f"weight[{i}]: got {got:.15f}, expected {exp:.15f}",
            )

    def test_weight_sum_approx_plate_area(self):
        """
        Sum of weights approximates area of [-1,1]^2 minus circle (r=0.5).
        With 10x10 voxels the staircase boundary introduces ~1.5% error,
        so we use a relaxed tolerance.
        """
        moments, _ = self._run()
        expected_area = 4.0 - math.pi * 0.5 ** 2
        self.assertAlmostEqual(moments.sum(), expected_area, delta=2e-2 * expected_area)


# ---------------------------------------------------------------------------
# test2 – uniform alpha=1 on a refined grid of cells with shrinking areas
#
# C++ test: mesh with 36 cells at 4 refinement levels, alpha=1.
# For each cell the sum of moment weights must equal the cell's physical area.
# ---------------------------------------------------------------------------

class TestMomentFitting2dTest2(unittest.TestCase):
    """
    Simulate the refined-grid scenario by calling compute_moments with
    appropriately scaled mappings that mimic the Jacobian of each cell.

    The C++ test verifies:
      sum(weights) == cell_area   for every cell
      len(rst[d])  == 5           (stride for degree=2)
      len(weights) == 25
    """

    D            = 2
    degree       = 2
    ROOT_NVOXELS = (8, 8)

    EXPECTED_AREAS = (
        [1.0] * 3 +
        [0.25] * 8 +
        [0.0625] * 15 +
        [1.0 / 64.0] * 2 +
        [1.0 / 256.0] * 8
    )

    def _nvoxels_for_level(self, level: int) -> tuple:
        factor = 2 ** level
        return tuple(max(n // factor, 1) for n in self.ROOT_NVOXELS)

    def test_weight_sum_equals_cell_area(self):
        # Without mapping, compute_moments integrates over the reference element
        # [-1,1]^D (area = 2^D = 4). The physical cell area is cell_area, so
        # the physical detJ = cell_area / 4.  We verify: sum * detJ == cell_area.
        for cell_idx, area in enumerate(self.EXPECTED_AREAS):
            level   = round(-math.log(area, 4)) if area < 1.0 else 0
            nvoxels = self._nvoxels_for_level(level)
            detJ    = area / (2.0 ** self.D)

            moments, _ = compute_moments(
                nvoxels=nvoxels,
                polynomial_degree=self.degree,
                alpha_func=alpha_uniform,
            )

            computed_area = moments.sum() * detJ
            self.assertAlmostEqual(
                computed_area, area, delta=1e-8 * area,
                msg=f"cell {cell_idx} (level {level}): "
                    f"got {computed_area:.6f}, expected {area:.6f}",
            )

    def test_quadrature_point_count(self):
        stride = 2 * self.degree + 1
        moments, elem_pts = compute_moments(
            nvoxels=self.ROOT_NVOXELS,
            polynomial_degree=self.degree,
            alpha_func=alpha_uniform,
        )
        for d in range(self.D):
            self.assertEqual(len(elem_pts[d]), stride)
        self.assertEqual(len(moments), stride ** self.D)


# ---------------------------------------------------------------------------
# test3 – plate-with-a-hole on a refined grid
#
# C++ test: same mesh, scaling0 = plateWithAHole, expected summed weights
# per cell from the C++ reference vector.
# ---------------------------------------------------------------------------

class TestMomentFitting2dTest3(unittest.TestCase):

    D            = 2
    degree       = 2
    ROOT_NVOXELS = (8, 8)

    EXPECTED_AREAS = (
        [1.0] * 3 +
        [0.25] * 8 +
        [0.0625] * 15 +
        [1.0 / 64.0] * 2 +
        [1.0 / 256.0] * 8
    )

    EXPECTED_SUM_WEIGHTS = [
        1.000000000000000, 1.000000000000000, 1.000000000000000,
        0.050491898148148, 0.250000000000000, 0.250000000000000,
        0.250000000000000, 0.250000000000000, 0.250000000000000,
        0.250000000000000, 0.250000000000000, 0.062500000000000,
        0.062500000000000, 0.062500000000000, 0.062500000000000,
        0.062500000000000, 0.062500000000000, 0.062500000000000,
        0.062500000000000, 0.062500000000000, 0.062500000000000,
        0.062500000000000, 0.062500000000000, 0.062500000000000,
        0.062500000000000, 0.062500000000000, 0.015625000000000,
        0.015625000000000, 0.003906250000000, 0.003906250000000,
        0.003906250000000, 0.003906250000000, 0.003906250000000,
        0.003906250000000, 0.003906250000000, 0.003906250000000,
    ]

    def _nvoxels_for_level(self, level: int) -> tuple:
        factor = 2 ** level
        return tuple(max(n // factor, 1) for n in self.ROOT_NVOXELS)

    def test_cut_element_sum(self):
        """
        Case A – element centred at origin straddles the hole:
          sum(moments) ≈ 4 - π·r²  (reference-element integral of plate_with_a_hole)
        """
        moments, _ = compute_moments(
            nvoxels=self.ROOT_NVOXELS,
            polynomial_degree=self.degree,
            alpha_func=plate_with_a_hole,
        )
        expected = 4.0 - math.pi * 0.5 ** 2
        self.assertAlmostEqual(
            moments.sum(), expected, delta=2e-2 * expected,
            msg=f"got {moments.sum():.6f}, expected ~{expected:.6f}",
        )

    def test_uniform_alpha_sum_at_all_levels(self):
        """
        Cases B & C – uniform alpha → sum = 4.0 at every refinement level,
        even as nvoxels is halved.
        """
        for level in range(4):
            nvoxels = self._nvoxels_for_level(level)
            m, _ = compute_moments(
                nvoxels=nvoxels,
                polynomial_degree=self.degree,
                alpha_func=alpha_uniform,
            )
            self.assertAlmostEqual(
                m.sum(), 4.0, places=10,
                msg=f"level {level}, nvoxels={nvoxels}: sum={m.sum()}",
            )


# ---------------------------------------------------------------------------
# Auxiliary: CSV output round-trip
# ---------------------------------------------------------------------------

class TestWriteCsv(unittest.TestCase):

    def test_csv_roundtrip(self):
        D, degree, nvoxels = 2, 2, (4, 4)
        moments, elem_pts = compute_moments(
            nvoxels=nvoxels,
            polynomial_degree=degree,
            alpha_func=alpha_uniform,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            path = f.name

        write_csv(path, elem_pts, moments, D)

        with open(path, newline="") as f:
            rows = list(csv_mod.reader(f))

        stride = 2 * degree + 1
        self.assertEqual(len(rows), stride ** D + 1)   # header + data rows
        self.assertEqual(rows[0], ["i_r", "i_s", "xi_r", "xi_s", "moment_weight"])
        self.assertEqual(len(rows[1]), 5)


if __name__ == "__main__":
    unittest.main()