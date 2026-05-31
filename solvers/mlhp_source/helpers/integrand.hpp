// NeuralMech extension to mlhp. License: See mlhp/LICENSE
//
// Custom physics integrands that live outside the pinned mlhp submodule so
// that the upstream source tree stays untouched. Built into the separate
// pymlhphelpers python module (see python/bindings.cpp).

#ifndef MLHP_HELPERS_INTEGRAND_HPP
#define MLHP_HELPERS_INTEGRAND_HPP

#include "mlhp/core/integrands.hpp"
#include "mlhp/core/spatial.hpp"

namespace mlhp::helpers
{

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
