// NeuralMech extension to mlhp. License: See mlhp/LICENSE
//
// Python bindings for the custom helper integrands. Compiled into the separate
// pymlhphelpers module. The wrapper types (ScalarFunctionWrapper, DomainIntegrand)
// are registered by pymlhpcore and shared across modules through pybind11's
// process-wide type registry, so the objects produced by `import mlhp` flow
// straight into these functions and the returned integrand back into mlhp.

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "src/python/pymlhpcore.hpp"

#include "integrand.hpp"
#include "j2plasticity.hpp"
#include "quadrature.hpp"

namespace mlhp::helpers
{

template<size_t D>
void bindDimension( pybind11::module& m )
{
    m.def( "helmholtzIntegrand", []( const bindings::ScalarFunctionWrapper<D>& wavenumber,
                                     const bindings::ScalarFunctionWrapper<D>& source )
        { return makeHelmholtzIntegrand<D>( wavenumber.get( ), source.get( ) ); },
        pybind11::arg( "wavenumber" ), pybind11::arg( "source" ),
        "Real-valued (undamped) Helmholtz integrand: laplacian(u) + k^2 u = -f." );

    auto zero = bindings::ScalarFunctionWrapper<D> { spatial::constantFunction<D>( 0.0 ) };

    m.def( "helmholtzIntegrand", []( const bindings::ScalarFunctionWrapper<D>& wavenumber,
                                     const bindings::ScalarFunctionWrapper<D>& damping,
                                     const bindings::ScalarFunctionWrapper<D>& sourceReal,
                                     const bindings::ScalarFunctionWrapper<D>& sourceImag )
        { return makeHelmholtzIntegrand<D>( wavenumber.get( ), damping.get( ),
                                            sourceReal.get( ), sourceImag.get( ) ); },
        pybind11::arg( "wavenumber" ), pybind11::arg( "damping" ),
        pybind11::arg( "source" ), pybind11::arg( "sourceImag" ) = zero,
        "Damped Helmholtz integrand: laplacian(u) + (k^2 + i eta) u = -f. Returns an "
        "unsymmetric two-field (real, imaginary) system." );

    m.def( "integratePartitionMatrices", []( const AbsBasis<D>& basis,
                                             const DomainIntegrand<D>& integrand,
                                             const AbsQuadrature<D>& quadrature,
                                             const bindings::QuadratureOrderDeterminorWrapper<D>& order,
                                             CellIndex icell )
        {
            auto flat = integratePartitionMatrices<D>( basis, integrand, quadrature, order.get( ), icell );

            auto locationMap = LocationMap { };
            basis.locationMap( icell, locationMap );

            auto ndof = static_cast<pybind11::ssize_t>( locationMap.size( ) );
            auto npart = ( ndof > 0 ) ? static_cast<pybind11::ssize_t>( flat.size( ) ) / ( ndof * ndof ) : 0;

            auto shape = { npart, ndof, ndof };
            auto strides = { static_cast<pybind11::ssize_t>( sizeof( double ) ) * ndof * ndof,
                             static_cast<pybind11::ssize_t>( sizeof( double ) ) * ndof,
                             static_cast<pybind11::ssize_t>( sizeof( double ) ) };

            return pybind11::array_t<double>( std::move( shape ), std::move( strides ), flat.data( ) );
        },
        pybind11::arg( "basis" ), pybind11::arg( "integrand" ), pybind11::arg( "quadrature" ),
        pybind11::arg( "order" ), pybind11::arg( "icell" ) = 0,
        "Per-partition element matrices for one element, shape [npartitions, ndof, ndof]. "
        "With gridQuadrature(nsubcells=...) each partition is one sub-cell, so this returns "
        "every sub-voxel's element stiffness in a single integration pass." );

    m.def( "voxelMomentFittingMatrix", []( std::array<size_t, D> nvoxels,
                                           size_t polynomialDegree )
        {
            auto flat = voxelMomentFittingMatrix<D>( nvoxels, polynomialDegree );

            auto nmf = static_cast<pybind11::ssize_t>( std::pow( 2 * polynomialDegree + 1, D ) );
            auto nvox = ( nmf > 0 ) ? static_cast<pybind11::ssize_t>( flat.size( ) ) / nmf : 0;

            auto shape = { nmf, nvox };
            auto strides = { static_cast<pybind11::ssize_t>( sizeof( double ) ) * nvox,
                             static_cast<pybind11::ssize_t>( sizeof( double ) ) };

            return pybind11::array_t<double>( std::move( shape ), std::move( strides ), flat.data( ) );
        },
        pybind11::arg( "nvoxels" ), pybind11::arg( "polynomialDegree" ),
        "Preintegrated voxel moment-fitting weight matrix M, shape [n_mf^D, nvox] with "
        "n_mf = 2*polynomialDegree+1. weights_e = detJ * (E_voxels @ M.T) gives the moment-fitting "
        "quadrature weights per element; pair with momentFittingPointMatrices to assemble stiffness." );

    m.def( "momentFittingPointMatrices", []( const AbsBasis<D>& basis,
                                             const DomainIntegrand<D>& integrand,
                                             size_t polynomialDegree,
                                             CellIndex icell )
        {
            auto flat = momentFittingPointMatrices<D>( basis, integrand, polynomialDegree, icell );

            auto locationMap = LocationMap { };
            basis.locationMap( icell, locationMap );

            auto ndof = static_cast<pybind11::ssize_t>( locationMap.size( ) );
            auto npts = ( ndof > 0 ) ? static_cast<pybind11::ssize_t>( flat.size( ) ) / ( ndof * ndof ) : 0;

            auto shape = { npts, ndof, ndof };
            auto strides = { static_cast<pybind11::ssize_t>( sizeof( double ) ) * ndof * ndof,
                             static_cast<pybind11::ssize_t>( sizeof( double ) ) * ndof,
                             static_cast<pybind11::ssize_t>( sizeof( double ) ) };

            return pybind11::array_t<double>( std::move( shape ), std::move( strides ), flat.data( ) );
        },
        pybind11::arg( "basis" ), pybind11::arg( "integrand" ),
        pybind11::arg( "polynomialDegree" ), pybind11::arg( "icell" ) = 0,
        "Unit-material per-Gauss-point element stiffness at the n_mf = 2*polynomialDegree+1 fixed "
        "moment-fitting points, shape [n_mf^D, ndof, ndof]. K_e = einsum('p,pij->ij', weights_e, this)." );

    m.def( "voxelMomentFittingQuadrature",
        []( std::array<size_t, D> nvoxels,
            const bindings::ScalarFunctionWrapper<D>& alpha,
            size_t polynomialDegree ) -> std::shared_ptr<AbsQuadrature<D>>
        {
            return std::make_shared<::voxelMomentFittingQuadrature<D>>(
                nvoxels, alpha.get( ), polynomialDegree );
        },
        pybind11::arg( "nvoxels" ), pybind11::arg( "alpha" ),
        pybind11::arg( "polynomialDegree" ) = 2,
        "Voxel moment-fitting quadrature. nvoxels: sub-voxels per element per axis. "
        "alpha: E-field spatial function. Quadrature weights exactly integrate polynomials "
        "of degree polynomialDegree weighted by alpha; set material E=1 when using this." );

    m.def( "voxelMomentFittingQuadrature",
        []( std::array<size_t, D> nvoxels,
            const bindings::ScalarFunctionWrapper<D>& alpha,
            size_t polynomialDegree,
            const bindings::ScalarFunctionWrapper<D>& scaling ) -> std::shared_ptr<AbsQuadrature<D>>
        {
            return std::make_shared<::voxelMomentFittingQuadrature<D>>(
                nvoxels, alpha.get( ), polynomialDegree, scaling.get( ) );
        },
        pybind11::arg( "nvoxels" ), pybind11::arg( "alpha" ),
        pybind11::arg( "polynomialDegree" ), pybind11::arg( "scaling" ),
        "Voxel moment-fitting quadrature with degradation scaling (e.g. phase-field g(s)). "
        "Weights encode alpha(x) * scaling(x) per voxel." );
}

void bindJ2Plasticity( pybind11::module& m )
{
    namespace py = pybind11;

    py::class_<AbsRefinedGridFunction<3>, std::shared_ptr<AbsRefinedGridFunction<3>>>( m, "J2History" )
        .def( "ncomponents", &AbsRefinedGridFunction<3>::ncomponents );

    m.def( "initialJ2History", []( std::shared_ptr<AbsHierarchicalGrid<3>> grid )
        { return makeInitialHistory<3>( grid, 13 ); }, py::arg( "grid" ),
        "Zero-initialized J2 history (6 stress + 6 backstress + 1 plastic strain) on a grid." );

    m.def( "j2Material", []( const bindings::ScalarFunctionWrapper<3>& youngsModulus,
                             const bindings::ScalarFunctionWrapper<3>& poissonRatio,
                             double yieldStress, double hardeningModulus, double hardeningRatio,
                             const SharedRefinedGridFunction<3>& history,
                             const bindings::ImplicitFunctionWrapper<3>& domain )
        { return makeJ2Material( youngsModulus.get( ), poissonRatio.get( ), yieldStress,
                                 hardeningModulus, hardeningRatio, history, domain.get( ) ); },
        py::arg( "youngsModulus" ), py::arg( "poissonRatio" ), py::arg( "yieldStress" ),
        py::arg( "hardeningModulus" ) = 0.0, py::arg( "hardeningRatio" ) = 0.5,
        py::arg( "history" ), py::arg( "domain" ),
        "Small-strain J2 constitutive equation; pass to staticDomainIntegrand." );

    m.def( "updateJ2History", []( const SharedRefinedGridFunction<3>& history,
                                  const MultilevelHpBasisSharedPtr<3>& basis,
                                  bindings::DoubleVector& dofs0, bindings::DoubleVector& dofs1,
                                  const std::shared_ptr<const KinematicEquation<3>>& kinematics,
                                  const bindings::ImplicitFunctionWrapper<3>& domain,
                                  const bindings::ScalarFunctionWrapper<3>& youngsModulus,
                                  const bindings::ScalarFunctionWrapper<3>& poissonRatio,
                                  double yieldStress, double hardeningModulus, double hardeningRatio,
                                  size_t quadratureOrder )
        { return updateJ2History( history, basis, dofs0.getShared( ), dofs1.getShared( ), kinematics,
                                  domain.get( ), youngsModulus.get( ), poissonRatio.get( ),
                                  yieldStress, hardeningModulus, hardeningRatio, quadratureOrder ); },
        py::arg( "history" ), py::arg( "basis" ), py::arg( "dofs0" ), py::arg( "dofs1" ),
        py::arg( "kinematics" ), py::arg( "domain" ), py::arg( "youngsModulus" ), py::arg( "poissonRatio" ),
        py::arg( "yieldStress" ), py::arg( "hardeningModulus" ), py::arg( "hardeningRatio" ),
        py::arg( "quadratureOrder" ),
        "Apply the return mapping over the strain increment dofs1-dofs0; returns updated history." );

    m.def( "evaluateJ2History", []( const SharedRefinedGridFunction<3>& history,
                                    const std::vector<std::array<double, 3>>& points )
        { return evaluateHistoryAtPoints( history, points ); },
        py::arg( "history" ), py::arg( "points" ),
        "Evaluate stored history at points; returns flat (npoints * 13) list." );
}

// Register all helper extensions into a module.  Called from the combined module
// entry (python/main.cpp), which also calls mlhp's own bind* functions, so the whole
// API ends up in a single `pymlhpcore` module.
void bindHelpers( pybind11::module& m )
{
    bindDimension<1>( m );
    bindDimension<2>( m );
    bindDimension<3>( m );
    bindJ2Plasticity( m );
}

} // namespace mlhp::helpers
