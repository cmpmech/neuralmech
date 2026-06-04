// NeuralMech extension to mlhp. License: See mlhp/LICENSE
//
// Custom physics integrands that live outside the pinned mlhp submodule so
// that the upstream source tree stays untouched. Built into the separate
// pymlhphelpers python module (see python/bindings.cpp).

#ifndef MLHP_HELPERS_INTEGRAND_HPP
#define MLHP_HELPERS_INTEGRAND_HPP

#include "mlhp/core/integrands.hpp"
#include "mlhp/core/basis.hpp"
#include "mlhp/core/quadrature.hpp"
#include "mlhp/core/spatial.hpp"

namespace mlhp::helpers
{

//! Integrate `integrand` over a single element `icell`, returning each quadrature PARTITION's
//! element matrix separately instead of summing them into one.  Mirrors integrateOnDomain's
//! element loop (mlhp/src/core/assembly.cpp) but resets the local target per partition and
//! snapshots its matrix target.  With gridQuadrature(nsubcells=...) one partition is one
//! sub-cell, so this yields every sub-voxel's element stiffness in a single integration pass.
//! Returns a flat row-major buffer of size npartitions * ndof * ndof (ndof = element dofs).
template<size_t D>
std::vector<double> integratePartitionMatrices( const AbsBasis<D>& basis,
                                                 const DomainIntegrand<D>& integrand,
                                                 const AbsQuadrature<D>& quadrature,
                                                 const QuadratureOrderDeterminor<D>& orderDeterminor,
                                                 CellIndex icell = 0 );

//! Preintegrated voxel moment-fitting weight matrix M (the element-independent part of
//! voxelMomentFittingQuadrature, helpers/quadrature.hpp). For a uniform Cartesian mesh the
//! only element-varying quantity in the moment computation is the per-sub-voxel material
//! value, so the map from voxel values to the n_mf = 2*polynomialDegree+1 fixed moment-fitting
//! Gauss weights is a single matrix reused for every element:
//!
//!     weights_e[p] = detJ * sum_v M[p, v] * E_voxel_e[v]
//!
//! M[p, v] = voxelDetJ * prod_d ( sum_{ip in voxel v_d} voxelWeight_d[ip] * L_{p_d}(voxelRst_d[ip]) )
//! with voxelDetJ = prod_d 1/nvoxels[d] and L the Lagrange cardinal polynomials nodal at the
//! n_mf Gauss points. Built from the ground-truth per-voxel summation (the scaling branch of
//! distribute), NOT the buggy tensor-product fast path. detJ and E are supplied in Python.
//!
//! Returns a flat row-major buffer of size n_mf^D * nvox (nvox = product(nvoxels)); the p index
//! runs row-major over {n_mf}^D and the v index row-major over nvoxels, matching the integration
//! point ordering of momentFittingPointMatrices below.
template<size_t D>
std::vector<double> voxelMomentFittingMatrix( std::array<size_t, D> nvoxels,
                                              size_t polynomialDegree );

//! Per-Gauss-point unit-material element stiffness B(x_p)^T C0 B(x_p) at the n_mf = 2*degree+1
//! fixed moment-fitting Gauss points of element `icell` (tensor product, row-major). Each point's
//! matrix is integrated with weight exactly 1 (no quadrature weight, no detJ) so that, combined
//! with voxelMomentFittingMatrix, the element stiffness is
//!
//!     K_e = sum_p weights_e[p] * Bmat[p]      (weights_e from M @ E_voxel, scaled by detJ)
//!
//! Mirrors integratePartitionMatrices but builds the fixed Gauss grid directly and forces unit
//! weight. Returns a flat row-major buffer of size n_mf^D * ndof * ndof (ndof = element dofs).
template<size_t D>
std::vector<double> momentFittingPointMatrices( const AbsBasis<D>& basis,
                                                 const DomainIntegrand<D>& integrand,
                                                 size_t polynomialDegree,
                                                 CellIndex icell = 0 );

//! Time-harmonic (frequency-domain) Helmholtz equation
//!
//!     laplacian(u) + k^2 u = -f      in Omega
//!
//! with wavenumber k(x) (= omega / c, angular frequency over wave speed) and
//! source f(x). The weak form, after integration by parts, gives the symmetric
//! (but indefinite for large k) element system
//!
//!     ( K - k^2 M ) u = F
//!
//! with stiffness K_ij = int grad(N_i) . grad(N_j) dOmega, mass M_ij =
//! int N_i N_j dOmega and load F_i = int N_i f dOmega.
//!
//! This version is purely real-valued (no damping). See the damped overload
//! below for the complex extension.
template<size_t D>
DomainIntegrand<D> makeHelmholtzIntegrand( const spatial::ScalarFunction<D>& wavenumber,
                                           const spatial::ScalarFunction<D>& source );

//! Damped time-harmonic Helmholtz equation (e^{-i omega t} convention)
//!
//!     laplacian(u) + ( k^2 + i eta ) u = -f      in Omega
//!
//! with damping eta(x) >= 0 and complex unknown u = u_re + i u_im and complex
//! source f = f_re + i f_im. Splitting into real and imaginary parts gives the
//! real two-field system (field 0 = real part, field 1 = imaginary part)
//!
//!     [ K - k^2 M      eta M    ] [ u_re ]   [ f_re ]
//!     [  -eta M      K - k^2 M  ] [ u_im ] = [ f_im ]
//!
//! with stiffness K, mass M as above. The +/- eta M coupling makes the system
//! non-symmetric, so this integrand assembles an unsymmetric matrix. Pass a
//! two-field basis (nfields = 2); eta = 0 recovers the undamped problem with
//! the imaginary part decoupling to zero.
template<size_t D>
DomainIntegrand<D> makeHelmholtzIntegrand( const spatial::ScalarFunction<D>& wavenumber,
                                           const spatial::ScalarFunction<D>& damping,
                                           const spatial::ScalarFunction<D>& sourceReal,
                                           const spatial::ScalarFunction<D>& sourceImag );

} // namespace mlhp::helpers

#endif // MLHP_HELPERS_INTEGRAND_HPP
