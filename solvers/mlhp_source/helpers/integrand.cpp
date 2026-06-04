// NeuralMech extension to mlhp. License: See mlhp/LICENSE

#include "integrand.hpp"

#include "mlhp/core/arrayfunctions.hpp"
#include "mlhp/core/basisevaluation.hpp"
#include "mlhp/core/dense.hpp"
#include "mlhp/core/mapping.hpp"
#include "mlhp/core/memory.hpp"
#include "mlhp/core/ndarray.hpp"
#include "mlhp/core/partitioning.hpp"
#include "mlhp/core/polynomials.hpp"
#include "mlhp/core/quadrature.hpp"
#include "mlhp/core/utilities.hpp"

namespace mlhp::helpers {

template <size_t D>
std::vector<double> integratePartitionMatrices(
    const AbsBasis<D> &basis, const DomainIntegrand<D> &integrand,
    const AbsQuadrature<D> &quadrature,
    const QuadratureOrderDeterminor<D> &orderDeterminor, CellIndex icell) {

  // locate matrix target in the integrand (assuming only one matrix integrand)
  auto matrixTarget = integrand.types.size();
  auto symmetric = true;

  for (size_t t = 0; t < integrand.types.size(); ++t) {
    if (integrand.types[t] == AssemblyType::SymmetricMatrix) {
      matrixTarget = t;
      symmetric = true;
      break;
    }
    if (integrand.types[t] == AssemblyType::UnsymmetricMatrix) {
      matrixTarget = t;
      symmetric = false;
      break;
    }
  }

  MLHP_CHECK(matrixTarget < integrand.types.size(),
             "Integrand has no matrix target.");

  // Per-thread style scratch (single element, so no parallel section needed).
  auto rst = CoordinateGrid<D>{};
  auto xyz = CoordinateList<D>{};
  auto weights = std::vector<double>{};
  auto locationMap = LocationMap{};
  auto shapes = BasisFunctionEvaluation<D>{};
  auto localTargets = AlignedDoubleVectors(integrand.types.size());
  auto quadratureCache = quadrature.initialize();
  auto basisCache = basis.createEvaluationCache();
  auto integrandCache = integrand.createCache(basis);

  // +1 & -1 to avoid underflow if DiffOrders has NoShapes = -1
  auto diffOrder =
      std::max(static_cast<size_t>(integrand.maxdiff) + 1, size_t{1}) - 1;

  auto maxdegrees =
      basis.prepareEvaluation(icell, diffOrder, shapes, basisCache);
  auto &mapping = basis.mapping(basisCache);
  auto npartitions = quadrature.partition(mapping, quadratureCache);
  auto accuracy = orderDeterminor(icell, maxdegrees);

  basis.locationMap(icell, locationMap);
  integrand.prepare(integrandCache, mapping, locationMap);

  auto ndof = locationMap.size();
  auto paddedSize = memory::paddedLength<double>(ndof);
  auto result = std::vector<double>(npartitions * ndof * ndof, 0.0);

  // Same as assembly.cpp's initializeLocalAssemblyTargets, but reset once per
  // partition.
  auto resetTargets = [&]() {
    for (size_t t = 0; t < integrand.types.size(); ++t) {
      auto type = static_cast<size_t>(integrand.types[t]);

      if (type == 0)
        localTargets[t].resize(1);
      if (type == 1)
        localTargets[t].resize(memory::paddedLength<double>(ndof));
      if (type == 2)
        localTargets[t].resize(
            linalg::denseMatrixStorageSize<linalg::UnsymmetricDenseMatrix>(
                ndof));
      if (type == 3)
        localTargets[t].resize(
            linalg::denseMatrixStorageSize<linalg::SymmetricDenseMatrix>(ndof));

      std::fill(localTargets[t].begin(), localTargets[t].end(), 0.0);
    }
  };

  for (size_t ipartition = 0; ipartition < npartitions; ++ipartition) {
    resetTargets();

    utilities::resize0(rst, xyz, weights);

    auto isGrid = quadrature.distribute(ipartition, accuracy, rst, xyz, weights,
                                        quadratureCache);

    if (isGrid) {
      basis.prepareGridEvaluation(rst, basisCache);

      nd::executeWithIndex(array::elementSizes(rst), [&](auto ijk, auto index) {
        basis.evaluateGridPoint(ijk, shapes, basisCache);
        integrand.evaluate(integrandCache, shapes, localTargets,
                           weights[index]);
      });
    } else {
      for (size_t index = 0; index < rst[0].size(); ++index) {
        basis.evaluateSinglePoint(
            array::extract(rst, array::makeSizes<D>(index)), shapes,
            basisCache);
        integrand.evaluate(integrandCache, shapes, localTargets,
                           weights[index]);
      }
    }

    // Snapshot this partition's dense element matrix into a full row-major
    // block.
    // result.data is the raw pointer to the start of the array
    auto *out = result.data() + ipartition * ndof * ndof;
    auto *data = localTargets[matrixTarget].data(); // contains results

    for (size_t i = 0; i < ndof; ++i) {
      for (size_t j = 0; j < ndof; ++j) {
        out[i * ndof + j] =
            symmetric
                ? linalg::indexDenseMatrix<linalg::SymmetricDenseMatrix>(
                      data, i, j, paddedSize)
                : linalg::indexDenseMatrix<linalg::UnsymmetricDenseMatrix>(
                      data, i, j, paddedSize);
      }
    }
  }

  return result;
}

template <size_t D>
std::vector<double> voxelMomentFittingMatrix(std::array<size_t, D> nvoxels,
                                             size_t polynomialDegree) {
  auto cache = QuadraturePointCache{};

  // The n_mf = 2p+1 fixed moment-fitting Gauss points per axis (Lagrange
  // nodes).
  auto nmf = 2 * polynomialDegree + 1;
  auto lagOrder = nmf - 1;
  auto orders = array::makeSizes<D>(nmf);

  auto rstPre = CoordinateGrid<D>{};
  tensorProductQuadrature(orders, rstPre, cache);

  // Sub-voxel Gauss points along each axis, concatenated voxel-by-voxel, so the
  // points of voxel v occupy the slice [v * voxelOrder, (v + 1) * voxelOrder).
  auto voxelOrder = static_cast<size_t>(std::ceil((lagOrder + 1.0) / 2.0));

  auto voxelRst = CoordinateGrid<D>{};
  auto voxelWeights = std::array<std::vector<double>, D>{};

  for (size_t d = 0; d < D; ++d) {
    for (size_t v = 0; v < nvoxels[d]; ++v) {
      auto points = CoordinateGrid<1>{};
      auto weights = std::vector<double>{};
      auto cache1 = QuadraturePointCache{};
      tensorProductQuadrature<1>({voxelOrder}, points, weights, cache1);

      auto splitter =
          makeCartesianMappingSplitter(CartesianMapping<1>{}, {nvoxels[d]});
      auto subMapping = splitter({v});
      subMapping.mapGrid(points);

      for (size_t i = 0; i < points[0].size(); ++i) {
        voxelRst[d].push_back(points[0][i]);
        voxelWeights[d].push_back(weights[i]);
      }
    }
  }

  // Lagrange cardinal polynomials nodal at rstPre, evaluated at every voxel
  // point (same call as helpers/quadrature.hpp): evalLag[d][lag * npts + ip].
  auto evalLag = std::array<std::vector<double>, D>{};

  for (size_t d = 0; d < D; ++d) {
    evalLag[d].resize(rstPre[d].size() * voxelRst[d].size(), 0.0);
    polynomial::lagrange(lagOrder, voxelRst[d].size(), rstPre[d].data(),
                         voxelRst[d].data(), evalLag[d].data());
  }

  // Per-axis per-voxel moments m1[d][lag * nV + v] = sum over THIS voxel's own
  // points of voxelWeight * L_lag. The moment matrix factorizes across axes.
  auto m1 = std::array<std::vector<double>, D>{};

  for (size_t d = 0; d < D; ++d) {
    auto nV = nvoxels[d];
    auto npts = voxelRst[d].size();
    m1[d].resize(nmf * nV, 0.0);

    for (size_t v = 0; v < nV; ++v) {
      for (size_t k = 0; k < voxelOrder; ++k) {
        auto ip = v * voxelOrder + k;
        auto w = voxelWeights[d][ip];

        for (size_t lag = 0; lag < nmf; ++lag) {
          m1[d][lag * nV + v] += evalLag[d][lag * npts + ip] * w;
        }
      }
    }
  }

  // Assemble M[p, voxel] = voxelDetJ * prod_d m1[d][p_d, voxel_d].
  auto voxelDetJ =
      array::product(array::divide(1.0, array::convert<double>(nvoxels)));
  auto nvox = array::product(nvoxels);
  auto result = std::vector<double>(array::product(orders) * nvox, 0.0);

  nd::executeWithIndex(nvoxels, [&](std::array<size_t, D> voxelIdx,
                                    size_t vFlat) {
    nd::executeWithIndex(orders, [&](std::array<size_t, D> p, size_t pFlat) {
      auto value = voxelDetJ;

      for (size_t d = 0; d < D; ++d) {
        value *= m1[d][p[d] * nvoxels[d] + voxelIdx[d]];
      }

      result[pFlat * nvox + vFlat] = value;
    });
  });

  return result;
}

template <size_t D>
std::vector<double>
momentFittingPointMatrices(const AbsBasis<D> &basis,
                           const DomainIntegrand<D> &integrand,
                           size_t polynomialDegree, CellIndex icell) {
  // Locate the (single) matrix target, as in integratePartitionMatrices.
  auto matrixTarget = integrand.types.size();
  auto symmetric = true;

  for (size_t t = 0; t < integrand.types.size(); ++t) {
    if (integrand.types[t] == AssemblyType::SymmetricMatrix) {
      matrixTarget = t;
      symmetric = true;
      break;
    }
    if (integrand.types[t] == AssemblyType::UnsymmetricMatrix) {
      matrixTarget = t;
      symmetric = false;
      break;
    }
  }

  MLHP_CHECK(matrixTarget < integrand.types.size(),
             "Integrand has no matrix target.");

  auto shapes = BasisFunctionEvaluation<D>{};
  auto localTargets = AlignedDoubleVectors(integrand.types.size());
  auto locationMap = LocationMap{};
  auto basisCache = basis.createEvaluationCache();
  auto integrandCache = integrand.createCache(basis);

  auto diffOrder =
      std::max(static_cast<size_t>(integrand.maxdiff) + 1, size_t{1}) - 1;

  basis.prepareEvaluation(icell, diffOrder, shapes, basisCache);
  auto &mapping = basis.mapping(basisCache);
  (void)mapping;

  basis.locationMap(icell, locationMap);
  integrand.prepare(integrandCache, mapping, locationMap);

  auto ndof = locationMap.size();
  auto paddedSize = memory::paddedLength<double>(ndof);

  auto resetTargets = [&]() {
    for (size_t t = 0; t < integrand.types.size(); ++t) {
      auto type = static_cast<size_t>(integrand.types[t]);

      if (type == 0)
        localTargets[t].resize(1);
      if (type == 1)
        localTargets[t].resize(memory::paddedLength<double>(ndof));
      if (type == 2)
        localTargets[t].resize(
            linalg::denseMatrixStorageSize<linalg::UnsymmetricDenseMatrix>(
                ndof));
      if (type == 3)
        localTargets[t].resize(
            linalg::denseMatrixStorageSize<linalg::SymmetricDenseMatrix>(ndof));

      std::fill(localTargets[t].begin(), localTargets[t].end(), 0.0);
    }
  };

  // The fixed moment-fitting Gauss grid (same points as
  // voxelMomentFittingMatrix).
  auto orders = array::makeSizes<D>(2 * polynomialDegree + 1);
  auto rst = CoordinateGrid<D>{};
  auto cache = QuadraturePointCache{};
  tensorProductQuadrature(orders, rst, cache);

  auto npoints = array::product(orders);
  auto result = std::vector<double>(npoints * ndof * ndof, 0.0);

  basis.prepareGridEvaluation(rst, basisCache);

  nd::executeWithIndex(array::elementSizes(rst), [&](auto ijk, auto index) {
    resetTargets();

    basis.evaluateGridPoint(ijk, shapes, basisCache);
    integrand.evaluate(integrandCache, shapes, localTargets, /*weight=*/1.0);

    auto *out = result.data() + index * ndof * ndof;
    auto *data = localTargets[matrixTarget].data();

    for (size_t i = 0; i < ndof; ++i) {
      for (size_t j = 0; j < ndof; ++j) {
        out[i * ndof + j] =
            symmetric
                ? linalg::indexDenseMatrix<linalg::SymmetricDenseMatrix>(
                      data, i, j, paddedSize)
                : linalg::indexDenseMatrix<linalg::UnsymmetricDenseMatrix>(
                      data, i, j, paddedSize);
      }
    }
  });

  return result;
}

template <size_t D>
DomainIntegrand<D>
makeHelmholtzIntegrand(const spatial::ScalarFunction<D> &wavenumber,
                       const spatial::ScalarFunction<D> &source) {
  auto evaluate = [=](const BasisFunctionEvaluation<D> &shapes,
                      AlignedDoubleVectors &targets, double weightDetJ) {
    auto k = wavenumber(shapes.xyz());
    auto f = source(shapes.xyz());

    auto massFactor = k * k * weightDetJ; // k^2 mass contribution
    auto loadFactor = f * weightDetJ;

    auto ndof = shapes.ndof();
    auto nblocks = shapes.nblocks();
    auto ndofpadded = shapes.ndofpadded();

    auto N = shapes.noalias(0, 0);
    auto dN = shapes.noalias(0, 1);

    linalg::symmetricElementLhs(
        targets[0].data(), ndof, nblocks, [=](size_t i, size_t j) {
          double laplacian = 0.0;

          for (size_t axis = 0; axis < D; ++axis) {
            laplacian += dN[axis * ndofpadded + i] * dN[axis * ndofpadded + j];
          }

          return laplacian * weightDetJ - N[i] * N[j] * massFactor;
        });

    linalg::elementRhs(targets[1].data(), ndof, nblocks,
                       [&](size_t i) { return N[i] * loadFactor; });
  };

  auto types =
      AssemblyTypeVector{AssemblyType::SymmetricMatrix, AssemblyType::Vector};

  return DomainIntegrand<D>(types, DiffOrders::FirstDerivatives, evaluate);
}

template <size_t D>
DomainIntegrand<D>
makeHelmholtzIntegrand(const spatial::ScalarFunction<D> &wavenumber,
                       const spatial::ScalarFunction<D> &damping,
                       const spatial::ScalarFunction<D> &sourceReal,
                       const spatial::ScalarFunction<D> &sourceImag) {
  auto evaluate = [=](const BasisFunctionEvaluation<D> &shapes,
                      AlignedDoubleVectors &targets, double weightDetJ) {
    MLHP_CHECK(shapes.nfields() == 2,
               "Damped Helmholtz integrand needs a two-field basis "
               "(field 0 = real part, field 1 = imaginary part).");

    auto xyz = shapes.xyz();

    auto stiffFactor = weightDetJ; // Laplacian
    auto massFactor =
        wavenumber(xyz) * wavenumber(xyz) * weightDetJ; // k^2 mass
    auto dampingFactor = damping(xyz) * weightDetJ;     // eta mass coupling

    auto ndofall = shapes.ndof();

    // field 0 = real part, field 1 = imaginary part (equal-order in practice)
    auto N0 = shapes.noalias(0, 0);
    auto dN0 = shapes.noalias(0, 1);
    auto N1 = shapes.noalias(1, 0);
    auto dN1 = shapes.noalias(1, 1);

    auto [ndof0, nblocks0, npad0] = shapes.sizes(0);
    auto [ndof1, nblocks1, npad1] = shapes.sizes(1);

    auto offset0 = fieldOffset(shapes, 0);
    auto offset1 = fieldOffset(shapes, 1);

    // Diagonal Helmholtz block H = K - k^2 M for shape functions Nr (rows) and
    // Nc (columns)
    auto helmholtzBlock = [=](auto Nr, auto dNr, size_t padr, auto Nc, auto dNc,
                              size_t padc) {
      return [=](size_t i, size_t j) {
        double laplacian = 0.0;

        for (size_t axis = 0; axis < D; ++axis) {
          laplacian += dNr[axis * padr + i] * dNc[axis * padc + j];
        }

        return laplacian * stiffFactor - Nr[i] * Nc[j] * massFactor;
      };
    };

    // Off-diagonal eta-mass coupling block with given sign
    auto couplingBlock = [=](auto Nr, auto Nc, double sign) {
      return [=](size_t i, size_t j) {
        return sign * Nr[i] * Nc[j] * dampingFactor;
      };
    };

    // The full matrix is unsymmetric, so every block is stored in full
    using Full = linalg::UnsymmetricDenseMatrix;

    auto data = targets[0].data();

    linalg::elementLhs<Full>(data, ndofall, offset0, ndof0, offset0, ndof0,
                             helmholtzBlock(N0, dN0, npad0, N0, dN0, npad0));
    linalg::elementLhs<Full>(data, ndofall, offset1, ndof1, offset1, ndof1,
                             helmholtzBlock(N1, dN1, npad1, N1, dN1, npad1));
    linalg::elementLhs<Full>(data, ndofall, offset0, ndof0, offset1, ndof1,
                             couplingBlock(N0, N1, +1.0));
    linalg::elementLhs<Full>(data, ndofall, offset1, ndof1, offset0, ndof0,
                             couplingBlock(N1, N0, -1.0));

    auto rhs = targets[1].data();

    linalg::elementRhs(rhs + offset0, ndof0, nblocks0, [&](size_t i) {
      return N0[i] * sourceReal(xyz) * weightDetJ;
    });
    linalg::elementRhs(rhs + offset1, ndof1, nblocks1, [&](size_t i) {
      return N1[i] * sourceImag(xyz) * weightDetJ;
    });
  };

  auto types =
      AssemblyTypeVector{AssemblyType::UnsymmetricMatrix, AssemblyType::Vector};

  return DomainIntegrand<D>(types, DiffOrders::FirstDerivatives, evaluate);
}

#define MLHP_INSTANTIATE_DIM(D)                                                \
  template DomainIntegrand<D> makeHelmholtzIntegrand(                          \
      const spatial::ScalarFunction<D> &wavenumber,                            \
      const spatial::ScalarFunction<D> &source);                               \
  template DomainIntegrand<D> makeHelmholtzIntegrand(                          \
      const spatial::ScalarFunction<D> &wavenumber,                            \
      const spatial::ScalarFunction<D> &damping,                               \
      const spatial::ScalarFunction<D> &sourceReal,                            \
      const spatial::ScalarFunction<D> &sourceImag);                           \
  template std::vector<double> integratePartitionMatrices(                     \
      const AbsBasis<D> &basis, const DomainIntegrand<D> &integrand,           \
      const AbsQuadrature<D> &quadrature,                                      \
      const QuadratureOrderDeterminor<D> &orderDeterminor, CellIndex icell);   \
  template std::vector<double> voxelMomentFittingMatrix(                       \
      std::array<size_t, D> nvoxels, size_t polynomialDegree);                 \
  template std::vector<double> momentFittingPointMatrices(                     \
      const AbsBasis<D> &basis, const DomainIntegrand<D> &integrand,           \
      size_t polynomialDegree, CellIndex icell);

MLHP_INSTANTIATE_DIM(1)
MLHP_INSTANTIATE_DIM(2)
MLHP_INSTANTIATE_DIM(3)

#undef MLHP_INSTANTIATE_DIM

} // namespace mlhp::helpers
