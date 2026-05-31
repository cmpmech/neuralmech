// NeuralMech extension to mlhp. License: See mlhp/LICENSE

#include "j2plasticity.hpp"

namespace mlhp::helpers
{

std::shared_ptr<ConstitutiveEquation<3>> makeJ2Material(
    const spatial::ScalarFunction<3>& youngsModulus, const spatial::ScalarFunction<3>& poissonRatio,
    double sigma0, double H, double beta,
    const SharedRefinedGridFunction<3>& history, const ImplicitFunction<3>& domain )
{
    static constexpr size_t D = 3;
    static constexpr size_t ncomponents = ( D * ( D + 1 ) ) / 2;

    using AnyCache = typename ConstitutiveEquation<D>::AnyCache;

    struct ThisCache { const AbsHierarchicalGrid<D>* grid; };

    auto create = []( const AbsBasis<D>& basis, const KinematicEquation<D>& ) -> AnyCache
    {
        auto grid = dynamic_cast<const AbsHierarchicalGrid<D>*>( &basis.mesh( ) );
        MLHP_CHECK( grid, "J2 plasticity needs a hierarchical grid to store history variables." );
        return ThisCache { .grid = grid };
    };

    auto evaluate = [=]( AnyCache& anyCache, const BasisFunctionEvaluation<D>& shapes,
                         std::span<const double> /* solutionGradient */, std::span<const double> strainIncrement,
                         std::span<double> stressTarget, std::span<double> tangent, double* strainEnergyDensity )
    {
        MLHP_CHECK( strainEnergyDensity == nullptr, "Strain energy density not implemented." );

        auto& cache = utilities::cast<ThisCache>( anyCache );
        auto data = history->template evaluate<13>( *cache.grid, shapes.elementIndex( ), shapes.rst( ), shapes.xyz( ) );
        auto j2 = J2PlasticityHelper( youngsModulus, poissonRatio, sigma0, H, beta, data, shapes.xyz( ), strainIncrement, domain );

        if( !tangent.empty( ) )
        {
            MLHP_CHECK( tangent.size( ) == ncomponents * ncomponents, "Inconsistent tangent target size." );
            j2.tangent( tangent );
        }
        if( !stressTarget.empty( ) )
        {
            auto stress = j2.stress( );
            MLHP_CHECK_DBG( stressTarget.size( ) == ncomponents, "Inconsistent stress target size." );
            std::copy( stress.begin( ), stress.end( ), stressTarget.begin( ) );
        }
    };

    return std::make_shared<ConstitutiveEquation<D>>( ConstitutiveEquation<D>
    {
        .create = std::move( create ), .evaluate = std::move( evaluate ),
        .ncomponents = ncomponents, .symmetric = true, .incremental = true, .name = "J2Plasticity"
    } );
}

namespace
{

// Lazy history function: evaluate strain increment from dofs and apply return mapping.
SharedRefinedGridFunction<3> makeHistoryUpdate(
    const SharedRefinedGridFunction<3>& oldHistory, const MultilevelHpBasisSharedPtr<3>& basis,
    const std::shared_ptr<const std::vector<double>>& dofs0, const std::shared_ptr<const std::vector<double>>& dofs1,
    const std::shared_ptr<const KinematicEquation<3>>& kinematics, const ImplicitFunction<3>& domain,
    const spatial::ScalarFunction<3>& youngsModulus, const spatial::ScalarFunction<3>& poissonRatio,
    double sigma0, double H, double beta )
{
    static constexpr size_t D = 3;

    struct Cache
    {
        BasisFunctionEvaluation<D> shapes;
        BasisEvaluationCache<D> basisCache;
        std::vector<DofIndex> locationMap;
        std::vector<double> tmp;
        typename KinematicEquation<D>::AnyCache kinematicsCache;
    };

    auto container = std::make_shared<utilities::ThreadLocalContainer<Cache>>( );
    for( size_t i = 0; i < container->data.size( ); ++i )
    {
        container->data[i].basisCache = basis->createEvaluationCache( );
        container->data[i].kinematicsCache = kinematics->create( *basis );
        container->data[i].tmp.resize( D * D + kinematics->ncomponents );
    }

    auto update = [=]( CellIndex historyCell, std::span<double> history0,
                       std::array<double, D> historyRst, std::array<double, D> xyz ) mutable
    {
        auto& cache = container->get( );
        oldHistory->evaluate( historyCell, history0, historyRst, xyz );

        auto evaluateStrain = [&]( const std::vector<double>& dofs )
        {
            auto backward = mesh::mapToOtherGrid( oldHistory->grid( ), basis->hierarchicalGrid( ), historyCell, historyRst );
            basis->prepareEvaluation( backward.first, 1, cache.shapes, cache.basisCache );
            basis->locationMap( backward.first, utilities::resize0( cache.locationMap ) );
            basis->evaluateSinglePoint( backward.second, cache.shapes, cache.basisCache );

            std::fill( cache.tmp.begin( ), cache.tmp.end( ), 0.0 );
            auto gradient = std::span( cache.tmp.data( ), D * D );
            auto strain = std::span( cache.tmp.data( ) + D * D, kinematics->ncomponents );

            evaluateSolutions( cache.shapes, cache.locationMap, dofs, gradient, 1 );
            kinematics->prepare( cache.kinematicsCache, basis->mapping( cache.basisCache ), cache.locationMap );
            kinematics->evaluate( cache.kinematicsCache, cache.shapes, gradient, strain, std::span<double> { } );
            return std::vector<double>( strain.begin( ), strain.end( ) );
        };

        auto strain0 = evaluateStrain( *dofs0 );
        auto strain1 = evaluateStrain( *dofs1 );
        for( size_t i = 0; i < strain1.size( ); ++i ) strain1[i] -= strain0[i];

        J2PlasticityHelper( youngsModulus, poissonRatio, sigma0, H, beta,
            history0, xyz, strain1, domain ).newHistory( history0 );
    };

    return makeRefinedGridFunction<D>( update, oldHistory->gridPtr( ), oldHistory->ncomponents( ) );
}

} // namespace

SharedRefinedGridFunction<3> updateJ2History(
    const SharedRefinedGridFunction<3>& history0, const MultilevelHpBasisSharedPtr<3>& basis,
    const std::shared_ptr<const std::vector<double>>& dofs0, const std::shared_ptr<const std::vector<double>>& dofs1,
    const std::shared_ptr<const KinematicEquation<3>>& kinematics, const ImplicitFunction<3>& domain,
    const spatial::ScalarFunction<3>& youngsModulus, const spatial::ScalarFunction<3>& poissonRatio,
    double sigma0, double H, double beta, size_t quadratureOrder )
{
    auto lazy = makeHistoryUpdate( history0, basis, dofs0, dofs1, kinematics, domain,
                                   youngsModulus, poissonRatio, sigma0, H, beta );

    // Materialize at Gauss points so the update is frozen for the next step
    return std::make_shared<GaussPointInterpolation<3>>( *lazy, history0->gridPtr( ), quadratureOrder );
}

std::vector<double> evaluateHistoryAtPoints( const SharedRefinedGridFunction<3>& history,
                                             const std::vector<std::array<double, 3>>& points )
{
    auto ncomponents = history->ncomponents( );
    auto result = std::vector<double>( points.size( ) * ncomponents, 0.0 );
    auto backwardMapping = history->grid( ).createBackwardMapping( );

    for( size_t i = 0; i < points.size( ); ++i )
    {
        auto mapped = backwardMapping->map( points[i] );
        if( mapped )
        {
            auto target = std::span( result.data( ) + i * ncomponents, ncomponents );
            history->evaluate( mapped->first, target, mapped->second, points[i] );
        }
    }
    return result;
}

} // namespace mlhp::helpers
