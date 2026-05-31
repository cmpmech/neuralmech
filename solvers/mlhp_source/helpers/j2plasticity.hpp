// NeuralMech extension to mlhp. License: See mlhp/LICENSE
//
// Small-strain von Mises (J2) plasticity with a return-mapping algorithm and
// Gauss-point history storage. Adapted from the upstream example
//   mlhp/examples/j2_pressurized_sphere_fcm.cpp
// (the J2 material lives only in that example, not in the linked core library),
// reduced to the fixed-mesh pieces needed to drive plasticity from python. The
// elasticity integrand/kinematics themselves come from the bound core API.

#ifndef MLHP_HELPERS_J2PLASTICITY_HPP
#define MLHP_HELPERS_J2PLASTICITY_HPP

#include "mlhp/core.hpp"

namespace mlhp::helpers
{

//! Stores history variables (here 13: 6 stress + 6 backstress + 1 effective
//! plastic strain) on a hierarchical grid and evaluates them at points.
template<size_t D>
class AbsRefinedGridFunction : public utilities::DefaultVirtualDestructor
{
    HierarchicalGridSharedPtr<D> grid_;
    size_t ncomponents_;

public:
    AbsRefinedGridFunction( const HierarchicalGridSharedPtr<D>& grid, size_t ncomponents ) :
        grid_ { grid }, ncomponents_ { ncomponents }
    { }

    const AbsHierarchicalGrid<D>& grid( ) const { return *grid_; }
    const HierarchicalGridSharedPtr<D> gridPtr( ) const { return grid_; }
    auto ncomponents( ) const { return ncomponents_; }

    template<size_t N>
    std::array<double, N> evaluate( const AbsHierarchicalGrid<D>& otherGrid,
                                    CellIndex otherCell, std::array<double, D> otherRst,
                                    std::array<double, D> xyz ) const
    {
        auto [historyCell, historyRst] = mesh::mapToOtherGrid( otherGrid, this->grid( ), otherCell, otherRst );
        return this->evaluate<N>( historyCell, historyRst, xyz );
    }

    void evaluate( const AbsHierarchicalGrid<D>& otherGrid, std::span<double> target,
                   CellIndex otherCell, std::array<double, D> otherRst, std::array<double, D> xyz ) const
    {
        auto [historyCell, historyRst] = mesh::mapToOtherGrid( otherGrid, this->grid( ), otherCell, otherRst );
        evaluateInternal( historyCell, target, historyRst, xyz );
    }

    template<size_t N>
    std::array<double, N> evaluate( CellIndex historyCell, std::array<double, D> rst,
                                    std::array<double, D> xyz ) const
    {
        MLHP_CHECK( N == ncomponents( ), "Inconsistent number of components." );
        auto target = std::array<double, N> { };
        evaluate( historyCell, target, rst, xyz );
        return target;
    }

    void evaluate( CellIndex historyCell, std::span<double> target,
                   std::array<double, D> rst, std::array<double, D> xyz ) const
    {
        evaluateInternal( historyCell, target, rst, xyz );
    }

private:
    virtual void evaluateInternal( CellIndex historyCell, std::span<double> target,
                                   std::array<double, D> rst, std::array<double, D> xyz ) const = 0;
};

template<size_t D>
using SharedRefinedGridFunction = std::shared_ptr<AbsRefinedGridFunction<D>>;

//! History materialized at Gauss-Legendre points per cell (nearest-point lookup).
template<size_t D>
class GaussPointInterpolation : public AbsRefinedGridFunction<D>
{
public:
    GaussPointInterpolation( std::shared_ptr<AbsHierarchicalGrid<D>> grid, size_t ncomponents, size_t quadratureOrder ) :
        AbsRefinedGridFunction<D>( grid, ncomponents ), quadratureOrder_ { quadratureOrder },
        points_ { gaussLegendrePoints( quadratureOrder )[0] }
    {
        auto generator = [this, n = size_t { 0 }]( ) mutable
        {
            auto n0 = n;
            n += this->ncomponents( ) * utilities::integerPow( quadratureOrder_, D );
            return n0;
        };
        offsets_.resize( this->grid( ).ncells( ) + 1 );
        std::generate( offsets_.begin( ), offsets_.end( ), generator );
        data_.resize( offsets_.back( ) );
    }

    GaussPointInterpolation( const AbsRefinedGridFunction<D>& oldHistory,
                             const HierarchicalGridSharedPtr<D>& newGrid, size_t quadratureOrder ) :
        GaussPointInterpolation( newGrid, oldHistory.ncomponents( ), quadratureOrder )
    {
        auto orders = array::makeSizes<D>( quadratureOrder );
        auto points = gaussLegendrePoints( quadratureOrder )[0];

        #pragma omp parallel
        {
            auto ncells = static_cast<std::int64_t>( newGrid->ncells( ) );
            auto mapping = newGrid->createMapping( );

            #pragma omp for
            for( std::int64_t ii = 0; ii < ncells; ++ii )
            {
                auto icell = static_cast<CellIndex>( ii );
                newGrid->prepareMapping( icell, mapping );

                nd::executeWithIndex( orders, [&]( std::array<size_t, D> ijk, size_t gaussPointIndex )
                {
                    auto newRst = std::array<double, D> { };
                    for( size_t axis = 0; axis < D; ++axis )
                    {
                        newRst[axis] = points[ijk[axis]];
                    }
                    auto [oldIndex, oldRst] = mesh::mapToOtherGrid( *newGrid, oldHistory.grid( ), icell, newRst );
                    auto itbegin = utilities::begin( data_, offsets_[icell] + gaussPointIndex * this->ncomponents( ) );
                    auto targetSpan = std::span( itbegin, oldHistory.ncomponents( ) );
                    oldHistory.evaluate( oldIndex, targetSpan, oldRst, mapping( newRst ) );
                } );
            }
        }
    }

    void evaluateInternal( CellIndex icell, std::span<double> target,
                           std::array<double, D> rst, std::array<double, D> ) const override
    {
        auto strides = nd::stridesFor( array::make<D>( quadratureOrder_ ) );
        auto index = size_t { 0 };
        for( size_t axis = 0; axis < D; ++axis )
        {
            auto index1D = utilities::findInterval( points_, rst[axis] );
            auto local = utilities::mapToLocal0( points_[index1D], points_[index1D + 1], rst[axis] );
            index += strides[axis] * ( local < 0.5 ? index1D : index1D + 1 );
        }
        auto begin = utilities::begin( data_, offsets_[icell] + index * this->ncomponents( ) );
        std::copy_n( begin, this->ncomponents( ), target.begin( ) );
    }

private:
    size_t quadratureOrder_;
    std::vector<double> points_;
    std::vector<size_t> offsets_;
    std::vector<double> data_;
};

template<size_t D>
using RefinedGridFunction = std::function<void( CellIndex historyCell, std::span<double> target,
                                                std::array<double, D> rst, std::array<double, D> xyz )>;

template<size_t D> inline
SharedRefinedGridFunction<D> makeRefinedGridFunction( const RefinedGridFunction<D>& function,
                                                      const HierarchicalGridSharedPtr<D>& grid, size_t ncomponents )
{
    struct GridFunction : public AbsRefinedGridFunction<D>
    {
        RefinedGridFunction<D> function_;
        GridFunction( const HierarchicalGridSharedPtr<D>& g, const RefinedGridFunction<D>& f, size_t n ) :
            AbsRefinedGridFunction<D>( g, n ), function_ { f } { }
        void evaluateInternal( CellIndex historyCell, std::span<double> target,
                               std::array<double, D> rst, std::array<double, D> xyz ) const override
        {
            function_( historyCell, target, rst, xyz );
        }
    };
    return std::make_shared<GridFunction>( grid, function, ncomponents );
}

//! Zero-initialized history (13 components) on the given grid.
template<size_t D> inline
SharedRefinedGridFunction<D> makeInitialHistory( std::shared_ptr<AbsHierarchicalGrid<D>> grid, size_t ncomponents )
{
    auto function = [=]( CellIndex, std::span<double> target, std::array<double, D>, std::array<double, D> )
    {
        std::fill( target.begin( ), target.end( ), 0.0 );
    };
    return makeRefinedGridFunction<D>( function, grid, ncomponents );
}

//! Radial-return mapping for small-strain J2 with linear isotropic/kinematic
//! hardening (beta in [0, 1]) and hardening modulus H.
class J2PlasticityHelper
{
    std::array<double, 6> stress_ { }, flowDirection_ { }, backstress0_ { };
    std::array<double, 6 * 6> elasticTangent_ { };
    double yieldFunction_ = 0.0, deltaLambda_ = 0.0, etaTrialNorm_ = 0.0, mu_ = 0.0, ep0_ = 0.0;
    bool isinside_ = false;
    double H_, beta_;

public:
    J2PlasticityHelper( const spatial::ScalarFunction<3>& youngsModulus,
                        const spatial::ScalarFunction<3>& poissonRatio,
                        double sigma0, double H, double beta,
                        std::span<const double> history0, std::array<double, 3> xyz,
                        std::span<const double> totalStrainIncrement, const ImplicitFunction<3>& domain ) :
        H_ { H }, beta_ { beta }
    {
        auto previousStress = std::array<double, 6> { };
        std::copy( history0.begin( ), history0.begin( ) + 6, previousStress.begin( ) );
        std::copy( history0.begin( ) + 6, history0.begin( ) + 12, backstress0_.begin( ) );
        ep0_ = history0[12];
        isinside_ = domain( xyz );

        auto nu = poissonRatio( xyz );
        auto tmp1 = ( 1.0 - 2.0 * nu );
        auto tmp2 = youngsModulus( xyz ) / ( ( 1.0 + nu ) * tmp1 );
        auto lambda = nu * tmp2;
        mu_ = 0.5 * tmp1 * tmp2;

        std::fill( elasticTangent_.begin( ), elasticTangent_.end( ), 0.0 );
        auto C = linalg::adapter( elasticTangent_, 6 );
        auto diagonal = lambda + 2.0 * mu_;
        C( 0, 0 ) = diagonal; C( 0, 1 ) = lambda;   C( 0, 2 ) = lambda;
        C( 1, 0 ) = lambda;   C( 1, 1 ) = diagonal; C( 1, 2 ) = lambda;
        C( 2, 0 ) = lambda;   C( 2, 1 ) = lambda;   C( 2, 2 ) = diagonal;
        C( 3, 3 ) = mu_; C( 4, 4 ) = mu_; C( 5, 5 ) = mu_;

        auto sigmaTrialIncrement = std::array<double, 6> { };
        linalg::mmproduct( elasticTangent_.data( ), totalStrainIncrement.data( ), sigmaTrialIncrement.data( ), 6, 6, 1 );

        auto sigmaTrial = previousStress + sigmaTrialIncrement;
        auto sigmaTrialTrace = sigmaTrial[0] + sigmaTrial[1] + sigmaTrial[2];
        auto unitTensor = std::array<double, 6> { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
        auto etaTrial = sigmaTrial - backstress0_ - 1.0 / 3.0 * sigmaTrialTrace * unitTensor;

        etaTrialNorm_ = std::sqrt( etaTrial[0] * etaTrial[0] + etaTrial[1] * etaTrial[1] + etaTrial[2] * etaTrial[2] +
                           2.0 * ( etaTrial[3] * etaTrial[3] + etaTrial[4] * etaTrial[4] + etaTrial[5] * etaTrial[5] ) );

        yieldFunction_ = etaTrialNorm_ - std::sqrt( 2.0 / 3.0 ) * ( sigma0 + ( 1.0 - beta_ ) * H_ * ep0_ );

        if( !isinside_ ) yieldFunction_ = -1.0;

        if( yieldFunction_ < 0.0 )
        {
            stress_ = sigmaTrial;
        }
        else
        {
            deltaLambda_ = yieldFunction_ / ( 2.0 * mu_ + 2.0 / 3.0 * H_ );
            flowDirection_ = etaTrial / etaTrialNorm_;
            stress_ = sigmaTrial - 2.0 * mu_ * deltaLambda_ * flowDirection_;
        }
    }

    auto stress( ) const { return stress_; }

    void newHistory( std::span<double> target )
    {
        auto factor = isinside_;
        auto backstress1 = backstress0_ * static_cast<double>( factor );
        auto ep1 = ep0_ * factor;

        if( yieldFunction_ >= 0.0 )
        {
            backstress1 = backstress0_ + ( 2.0 / 3.0 ) * beta_ * H_ * deltaLambda_ * flowDirection_;
            ep1 = ep0_ + std::sqrt( 2.0 / 3.0 ) * deltaLambda_;
        }
        std::copy( stress_.begin( ), stress_.end( ), target.begin( ) );
        std::copy( backstress1.begin( ), backstress1.end( ), target.begin( ) + 6 );
        target[12] = ep1;
    }

    void tangent( std::span<double> target ) const
    {
        if( yieldFunction_ < 0.0 )
        {
            std::copy( elasticTangent_.begin( ), elasticTangent_.end( ), target.begin( ) );
        }
        else
        {
            auto c1 = 4.0 * mu_ * mu_ / ( 2.0 * mu_ + 2.0 / 3.0 * H_ );
            auto c2 = 4.0 * mu_ * mu_ * deltaLambda_ / etaTrialNorm_;
            auto C = linalg::adapter( elasticTangent_, 6 );
            auto Dalg = linalg::adapter( target, 6 );
            for( size_t i = 0; i < 6; i++ )
                for( size_t j = 0; j < 6; j++ )
                    Dalg( i, j ) = C( i, j ) - ( c1 - c2 ) * flowDirection_[i] * flowDirection_[j];
            for( size_t i = 0; i < 3; ++i )
            {
                for( size_t j = 0; j < 3; ++j ) Dalg( i, j ) -= -1.0 / 3.0 * c2;
                Dalg( i + 0, i + 0 ) -= c2;
                Dalg( i + 3, i + 3 ) -= c2 / 2.0;
            }
        }
    }
};

//! J2 constitutive equation reading/using history; plug into staticDomainIntegrand.
std::shared_ptr<ConstitutiveEquation<3>> makeJ2Material(
    const spatial::ScalarFunction<3>& youngsModulus, const spatial::ScalarFunction<3>& poissonRatio,
    double sigma0, double H, double beta,
    const SharedRefinedGridFunction<3>& history, const ImplicitFunction<3>& domain );

//! Apply the return mapping over the strain increment dofs1-dofs0 and return the
//! updated history, materialized at Gauss points (call once per converged step).
SharedRefinedGridFunction<3> updateJ2History(
    const SharedRefinedGridFunction<3>& history0,
    const MultilevelHpBasisSharedPtr<3>& basis,
    const std::shared_ptr<const std::vector<double>>& dofs0,
    const std::shared_ptr<const std::vector<double>>& dofs1,
    const std::shared_ptr<const KinematicEquation<3>>& kinematics,
    const ImplicitFunction<3>& domain,
    const spatial::ScalarFunction<3>& youngsModulus, const spatial::ScalarFunction<3>& poissonRatio,
    double sigma0, double H, double beta, size_t quadratureOrder );

//! Evaluate stored history at global points; returns ncomponents values per point.
std::vector<double> evaluateHistoryAtPoints( const SharedRefinedGridFunction<3>& history,
                                             const std::vector<std::array<double, 3>>& points );

} // namespace mlhp::helpers

#endif // MLHP_HELPERS_J2PLASTICITY_HPP
