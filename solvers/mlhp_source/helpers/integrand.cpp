// NeuralMech extension to mlhp. License: See mlhp/LICENSE

#include "integrand.hpp"

#include "mlhp/core/basisevaluation.hpp"
#include "mlhp/core/dense.hpp"

namespace mlhp::helpers
{

template<size_t D>
DomainIntegrand<D> makeHelmholtzIntegrand( const spatial::ScalarFunction<D>& wavenumber,
                                           const spatial::ScalarFunction<D>& source )
{
    auto evaluate = [=]( const BasisFunctionEvaluation<D>& shapes,
                         AlignedDoubleVectors& targets,
                         double weightDetJ )
    {
        auto k = wavenumber( shapes.xyz( ) );
        auto f = source( shapes.xyz( ) );

        auto massFactor = k * k * weightDetJ; // k^2 mass contribution
        auto loadFactor = f * weightDetJ;

        auto ndof = shapes.ndof( );
        auto nblocks = shapes.nblocks( );
        auto ndofpadded = shapes.ndofpadded( );

        auto N = shapes.noalias( 0, 0 );
        auto dN = shapes.noalias( 0, 1 );

        linalg::symmetricElementLhs( targets[0].data( ), ndof, nblocks, [=]( size_t i, size_t j )
        {
            double laplacian = 0.0;

            for( size_t axis = 0; axis < D; ++axis )
            {
                laplacian += dN[axis * ndofpadded + i] * dN[axis * ndofpadded + j];
            }

            return laplacian * weightDetJ - N[i] * N[j] * massFactor;
        } );

        linalg::elementRhs( targets[1].data( ), ndof, nblocks, [&]( size_t i )
        {
            return N[i] * loadFactor;
        } );
    };

    auto types = AssemblyTypeVector { AssemblyType::SymmetricMatrix, AssemblyType::Vector };

    return DomainIntegrand<D>( types, DiffOrders::FirstDerivatives, evaluate );
}

template<size_t D>
DomainIntegrand<D> makeHelmholtzIntegrand( const spatial::ScalarFunction<D>& wavenumber,
                                           const spatial::ScalarFunction<D>& damping,
                                           const spatial::ScalarFunction<D>& sourceReal,
                                           const spatial::ScalarFunction<D>& sourceImag )
{
    auto evaluate = [=]( const BasisFunctionEvaluation<D>& shapes,
                         AlignedDoubleVectors& targets,
                         double weightDetJ )
    {
        MLHP_CHECK( shapes.nfields( ) == 2, "Damped Helmholtz integrand needs a two-field basis "
                                            "(field 0 = real part, field 1 = imaginary part)." );

        auto xyz = shapes.xyz( );

        auto stiffFactor = weightDetJ;                          // Laplacian
        auto massFactor = wavenumber( xyz ) * wavenumber( xyz ) * weightDetJ; // k^2 mass
        auto dampingFactor = damping( xyz ) * weightDetJ;       // eta mass coupling

        auto ndofall = shapes.ndof( );

        // field 0 = real part, field 1 = imaginary part (equal-order in practice)
        auto N0 = shapes.noalias( 0, 0 );
        auto dN0 = shapes.noalias( 0, 1 );
        auto N1 = shapes.noalias( 1, 0 );
        auto dN1 = shapes.noalias( 1, 1 );

        auto [ndof0, nblocks0, npad0] = shapes.sizes( 0 );
        auto [ndof1, nblocks1, npad1] = shapes.sizes( 1 );

        auto offset0 = fieldOffset( shapes, 0 );
        auto offset1 = fieldOffset( shapes, 1 );

        // Diagonal Helmholtz block H = K - k^2 M for shape functions Nr (rows) and Nc (columns)
        auto helmholtzBlock = [=]( auto Nr, auto dNr, size_t padr, auto Nc, auto dNc, size_t padc )
        {
            return [=]( size_t i, size_t j )
            {
                double laplacian = 0.0;

                for( size_t axis = 0; axis < D; ++axis )
                {
                    laplacian += dNr[axis * padr + i] * dNc[axis * padc + j];
                }

                return laplacian * stiffFactor - Nr[i] * Nc[j] * massFactor;
            };
        };

        // Off-diagonal eta-mass coupling block with given sign
        auto couplingBlock = [=]( auto Nr, auto Nc, double sign )
        {
            return [=]( size_t i, size_t j ) { return sign * Nr[i] * Nc[j] * dampingFactor; };
        };

        // The full matrix is unsymmetric, so every block is stored in full
        using Full = linalg::UnsymmetricDenseMatrix;

        auto data = targets[0].data( );

        linalg::elementLhs<Full>( data, ndofall, offset0, ndof0, offset0, ndof0,
            helmholtzBlock( N0, dN0, npad0, N0, dN0, npad0 ) );
        linalg::elementLhs<Full>( data, ndofall, offset1, ndof1, offset1, ndof1,
            helmholtzBlock( N1, dN1, npad1, N1, dN1, npad1 ) );
        linalg::elementLhs<Full>( data, ndofall, offset0, ndof0, offset1, ndof1,
            couplingBlock( N0, N1, +1.0 ) );
        linalg::elementLhs<Full>( data, ndofall, offset1, ndof1, offset0, ndof0,
            couplingBlock( N1, N0, -1.0 ) );

        auto rhs = targets[1].data( );

        linalg::elementRhs( rhs + offset0, ndof0, nblocks0,
            [&]( size_t i ) { return N0[i] * sourceReal( xyz ) * weightDetJ; } );
        linalg::elementRhs( rhs + offset1, ndof1, nblocks1,
            [&]( size_t i ) { return N1[i] * sourceImag( xyz ) * weightDetJ; } );
    };

    auto types = AssemblyTypeVector { AssemblyType::UnsymmetricMatrix, AssemblyType::Vector };

    return DomainIntegrand<D>( types, DiffOrders::FirstDerivatives, evaluate );
}

#define MLHP_INSTANTIATE_DIM( D )                                                     \
    template DomainIntegrand<D> makeHelmholtzIntegrand(                               \
        const spatial::ScalarFunction<D>& wavenumber,                                \
        const spatial::ScalarFunction<D>& source );                                  \
    template DomainIntegrand<D> makeHelmholtzIntegrand(                               \
        const spatial::ScalarFunction<D>& wavenumber,                                \
        const spatial::ScalarFunction<D>& damping,                                   \
        const spatial::ScalarFunction<D>& sourceReal,                                \
        const spatial::ScalarFunction<D>& sourceImag );

MLHP_INSTANTIATE_DIM( 1 )
MLHP_INSTANTIATE_DIM( 2 )
MLHP_INSTANTIATE_DIM( 3 )

#undef MLHP_INSTANTIATE_DIM

} // namespace mlhp::helpers
