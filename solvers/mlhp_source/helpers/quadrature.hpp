#include "mlhp/core.hpp"
#include <chrono>
#include <filesystem>

using namespace mlhp;

template<size_t D>
void tensorProductQuadratures( std::array<size_t, D> orders,
                               CoordinateGrid<D>& rst,
                               CoordinateGrid<D>& weights )
{
    for( size_t axis = 0; axis < D; ++axis )
    {
        auto integrationPoints = gaussLegendrePoints( orders[axis] );
        rst[axis] = integrationPoints[0];
        weights[axis] = integrationPoints[1];
        // std::cout << axis << " : " << integrationPoints[axis] << std::endl;
    }
}


template<size_t D>
class voxelMomentFittingQuadrature final : public AbsQuadrature<D> // Standard Quadrature inherits from Abstract Quadrature Class
{
public:

    voxelMomentFittingQuadrature( std::array<size_t, D> nvoxels,
                                  spatial::ScalarFunction<D> alpha,
                                  size_t polynomilaDegree,
                                  spatial::ScalarFunction<D> scaling);

    voxelMomentFittingQuadrature( std::array<size_t, D> nvoxels,
                                  spatial::ScalarFunction<D> alpha,
                                  size_t polynomilaDegree );

    voxelMomentFittingQuadrature(   const AbsHierarchicalGrid<D>& grid,
                                    std::array<size_t, D> rootVoxel,
                                    spatial::ScalarFunction<D> alpha,
                                    size_t polynomilaDegree,
                                    spatial::ScalarFunction<D> scaling,
                                    size_t refinementLevel );

    voxelMomentFittingQuadrature(   const AbsHierarchicalGrid<D>& grid,
                                    std::array<size_t, D> rootVoxel,
                                    spatial::ScalarFunction<D> alpha,
                                    size_t polynomilaDegree,
                                    size_t refinementLevel );

    voxelMomentFittingQuadrature(   const AbsHierarchicalGrid<D>& grid,
                                std::array<size_t, D> rootVoxel,
                                spatial::ScalarFunction<D> alpha,
                                size_t polynomilaDegree );

    QuadratureCache<D> initialize( ) const override; // Overriding function in base class AbsQuadrature<D>

    size_t partition( const MeshMapping<D>& mapping,
                      QuadratureCache<D>& anyCache ) const override; // Not entirely clear yet to me

    bool distribute( size_t ipartition,
                     std::array<size_t, D> orders,
                     CoordinateGrid<D>& rst,
                     CoordinateList<D>& xyzList,
                     std::vector<double>& weights,
                     QuadratureCache<D>& anyCache ) const override;

private:
    struct Cache;
    std::function<std::array<size_t, D>( CellIndex icell )> nvoxels_;
    spatial::ScalarFunction<D> alpha_;
    size_t polynomialDegree_;
    std::optional<spatial::ScalarFunction<D>> scaling_;
    std::optional<size_t> refinementLevels_;
    const AbsHierarchicalGrid<D>* grid_ = nullptr;
    };


template<size_t D>
struct voxelMomentFittingQuadrature<D>::Cache
{
    QuadraturePointCache quadrature;
    std::array<size_t, D> nvoxels;
    const AbsMapping<D>* mapping;
    CellIndex icell;

    // data per refinement levels
    std::vector<CoordinateGrid<D>> voxelRstPerLevel;
    std::vector<std::array<std::vector<double>, D>> voxelWeightsPerLevel;
    std::vector<std::array<std::vector<double>, D>> lagrangeEvalPerLevel;


    /*
    std::array<std::vector<double>, D> lagrangeEval;
    CoordinateGrid<D> voxelRst;
    std::array<std::vector<double>, D> voxelWeights;
    */

};



// base constructor
template<size_t D>
voxelMomentFittingQuadrature<D>::voxelMomentFittingQuadrature(
    std::array<size_t, D> nvoxels,
    spatial::ScalarFunction<D> alpha,
    size_t polynomilaDegree,
    spatial::ScalarFunction<D> scaling ) :

    nvoxels_ { [=]( CellIndex ) noexcept { return nvoxels; } },
    alpha_( alpha ),
    polynomialDegree_( polynomilaDegree ),
    // scaling should be renamed...maybe solution field?
    scaling_( scaling ),
    refinementLevels_( std::nullopt )

{ }


template<size_t D>
voxelMomentFittingQuadrature<D>::voxelMomentFittingQuadrature(
    std::array<size_t, D> nvoxels,
    spatial::ScalarFunction<D> alpha,
    size_t polynomilaDegree ) :

    nvoxels_ { [=]( CellIndex ) noexcept { return nvoxels; } },
    alpha_( alpha ),
    polynomialDegree_( polynomilaDegree ),
    // scaling should be renamed...maybe solution field?
    scaling_( std::nullopt ),
    refinementLevels_( std::nullopt )

{ }

// refinement constructor
template<size_t D>
voxelMomentFittingQuadrature<D>::voxelMomentFittingQuadrature(
    const AbsHierarchicalGrid<D>& grid,
    std::array<size_t, D> rootVoxel,
    spatial::ScalarFunction<D> alpha,
    size_t polynomilaDegree,
    spatial::ScalarFunction<D> scaling,
    size_t refinementLevel ) :

    alpha_( alpha ),
    polynomialDegree_( polynomilaDegree ),
    scaling_( scaling ),
    refinementLevels_( refinementLevel ),
    grid_( &grid )

{
    nvoxels_ = [=, &grid]( CellIndex icell )
    {
        auto factor = utilities::binaryPow<size_t>( grid.refinementLevel( grid.fullIndex( icell ) ) );
        auto nvoxels = array::maxArray( array::divide( rootVoxel, factor ), array::makeSizes<D>( 1 ) );
        return nvoxels;
    };
}


template<size_t D>
voxelMomentFittingQuadrature<D>::voxelMomentFittingQuadrature(
    const AbsHierarchicalGrid<D>& grid,
    std::array<size_t, D> rootVoxel,
    spatial::ScalarFunction<D> alpha,
    size_t polynomilaDegree ) :

    alpha_( alpha ),
    polynomialDegree_( polynomilaDegree ),
    scaling_( std::nullopt ),
    refinementLevels_( std::nullopt  ),
    grid_( &grid )

{
    nvoxels_ = [=, &grid]( CellIndex icell )
    {
        auto factor = utilities::binaryPow<size_t>( grid.refinementLevel( grid.fullIndex( icell ) ) );
        auto nvoxels = array::maxArray( array::divide( rootVoxel, factor ), array::makeSizes<D>( 1 ) );
        return nvoxels;
    };
}


template<size_t D>
voxelMomentFittingQuadrature<D>::voxelMomentFittingQuadrature(
    const AbsHierarchicalGrid<D>& grid,
    std::array<size_t, D> rootVoxel,
    spatial::ScalarFunction<D> alpha,
    size_t polynomilaDegree,
    size_t refinementLevel ) :

    alpha_( alpha ),
    polynomialDegree_( polynomilaDegree ),
    scaling_( std::nullopt ),
    refinementLevels_( refinementLevel  ),
    grid_( &grid )

{
    nvoxels_ = [=, &grid]( CellIndex icell )
    {
        auto factor = utilities::binaryPow<size_t>( grid.refinementLevel( grid.fullIndex( icell ) ) );
        auto nvoxels = array::maxArray( array::divide( rootVoxel, factor ), array::makeSizes<D>( 1 ) );
        return nvoxels;
    };
}


template<size_t D>
QuadratureCache<D> voxelMomentFittingQuadrature<D>::initialize( ) const
{
    Cache cache { };

    if ( refinementLevels_ )
    {
        cache.voxelRstPerLevel.resize( refinementLevels_.value( ) );
        cache.voxelWeightsPerLevel.resize( refinementLevels_.value( ) );
        cache.lagrangeEvalPerLevel.resize( refinementLevels_.value( ) );

        // go through each refinement level and construct the lagrange evaluation for different voxel levels
        for ( int level = 0; level < refinementLevels_;  level++ )
        {
            auto factor = utilities::binaryPow<size_t>( level );
            auto nvoxels = array::maxArray( array::divide( nvoxels_( CellIndex( 0 ) ), factor ), array::makeSizes<D>( 1 ) );

            CoordinateGrid<D> rstPre;
            std::vector<double> weightsPre;
            std::array<size_t, D> n_mfPre;
            std::fill( n_mfPre.begin(), n_mfPre.end( ), 2 * polynomialDegree_ + 1 );

            // Integration points on the element (for integration of stiffness matrix)
            tensorProductQuadrature( n_mfPre, rstPre, weightsPre, cache.quadrature );
            // reset weights
            std::fill( weightsPre.begin(), weightsPre.end(), 1.0);


            // integration order on for the lagrange polynomials
            auto lagOrderPre = ( 2 * polynomialDegree_ + 1 ) - 1;
            auto voxelOrdersPre = array::makeSizes<D>( static_cast<size_t>( std::ceil( ( lagOrderPre + 1.0 ) / 2.0 ) ) );


            // Loop over each voxel
            // 1. generating integration points
            // 2. mapping the integration point to the element domain
            // 3. multiplying the voxel weight
            // 4. assembling the voxel interation points to the full integration point list (of voxels)

            CoordinateGrid<D> voxelRst;
            CoordinateList<D> voxelXyz;
            std::array<std::vector<double>, D> voxelWeights;
            auto voxelDetJ = 2.0 / nvoxels_( CellIndex( 0 ) )[0]; // assuming constant number of voxels in each direction!

            for(size_t d = 0; d < D; ++d )
            {
                CoordinateGrid<1> points;
                std::vector<double> weights;

                for(size_t voxelCounter = 0; voxelCounter < nvoxels[d]; ++voxelCounter)
                {
                    // Generate 1D points
                    QuadraturePointCache quadrature;
                    tensorProductQuadrature( { voxelOrdersPre[0] }, points, weights, quadrature );

                    // Generate mapping for the points
                    auto cellMapping = makeCartesianMappingSplitter( CartesianMapping<1>( ), { nvoxels[d] } );
                    auto subvoxelMapping = cellMapping( { voxelCounter } );

                    // map the points to the element
                    subvoxelMapping.mapGrid( points );

                    for(size_t i = 0; i < points[0].size( ); i++)
                    {
                        voxelRst[d].push_back( points[0][i] );
                        voxelWeights[d].push_back( weights[i]  );

                    } // collection ip  / voxels
                } // voxel loop
            } // axis loop

            // store voxel integration points and weights in cache
            cache.voxelRstPerLevel[level] = voxelRst;
            cache.voxelWeightsPerLevel[level] = voxelWeights;

            // now generate lagrange polynomials with points on the element integration points and evaluate these lagrange polynomials on each voxel integration point
            std::array<std::vector<double>, D> evaluation; // placeholder to store the evaluations

            for ( size_t d = 0; d < D; ++d ) // for each dimension
            {
                evaluation[d].resize( rstPre[d].size( ) * voxelRst[d].size( ), 0.0 );

                polynomial::lagrange(   lagOrderPre, // order of the lagrange polynomial
                                        voxelRst[d].size( ), // Number of evaluation points
                                        rstPre[d].data( ), // Element integration points (Stützstellen)
                                        voxelRst[d].data( ), // Auswertungstellen
                                        evaluation[d].data( ) );
            }

            cache.lagrangeEvalPerLevel[level] = evaluation;

        } // loop over refinement levels

    } // if refined elements
    else
    {
        // Setting vector length to 1
        cache.voxelRstPerLevel.resize( 1 );
        cache.voxelWeightsPerLevel.resize( 1 );
        cache.lagrangeEvalPerLevel.resize( 1 );

        // using root level voxels
        cache.nvoxels = nvoxels_( CellIndex( 0 ) );

        CoordinateGrid<D> rstPre;
        std::vector<double> weightsPre;
        std::array<size_t, D> n_mfPre;
        std::fill( n_mfPre.begin(), n_mfPre.end( ), 2 * polynomialDegree_ + 1 );

        // Integration points on the element (for integration of stiffness matrix)
        tensorProductQuadrature( n_mfPre, rstPre, weightsPre, cache.quadrature );
        // reset weights
        std::fill( weightsPre.begin(), weightsPre.end(), 1.0);


        // integration order on for the lagrange polynomials
        auto lagOrderPre = ( 2 * polynomialDegree_ + 1 ) - 1;
        auto voxelOrdersPre = array::makeSizes<D>( static_cast<size_t>( std::ceil( ( lagOrderPre + 1.0 ) / 2.0 ) ) );


        // Loop over each voxel
        // 1. generating integration points
        // 2. mapping the integration point to the element domain
        // 3. multiplying the voxel weight
        // 4. assembling the voxel interation points to the full integration point list (of voxels)

        CoordinateGrid<D> voxelRst;
        CoordinateList<D> voxelXyz;
        std::array<std::vector<double>, D> voxelWeights;
        auto voxelDetJ = 2.0 / nvoxels_( CellIndex( 0 ) )[0]; // assuming constant number of voxels in each direction!

        for(size_t d = 0; d < D; ++d )
        {
            CoordinateGrid<1> points;
            std::vector<double> weights;

            for(size_t voxelCounter = 0; voxelCounter < nvoxels_( CellIndex( 0 ) )[d]; ++voxelCounter)
            {
                // Generate 1D points
                QuadraturePointCache quadrature;
                tensorProductQuadrature( { voxelOrdersPre[0] }, points, weights, quadrature );

                // Generate mapping for the points
                auto cellMapping = makeCartesianMappingSplitter( CartesianMapping<1>( ), { nvoxels_( CellIndex( 0 ) )[0] } );
                auto subvoxelMapping = cellMapping( { voxelCounter } );

                // map the points to the element
                subvoxelMapping.mapGrid( points );

                for(size_t i = 0; i < points[0].size( ); i++)
                {
                    voxelRst[d].push_back( points[0][i] );
                    voxelWeights[d].push_back( weights[i]  );

                } // collection ip  / voxels
            } // voxel loop
        } // axis loop

        // store voxel integration points and weights in cache
        cache.voxelRstPerLevel[0] = voxelRst;
        cache.voxelWeightsPerLevel[0] = voxelWeights;

        // now generate lagrange polynomials with points on the element integration points and evaluate these lagrange polynomials on each voxel integration point
        std::array<std::vector<double>, D> evaluation; // placeholder to store the evaluations

        for ( size_t d = 0; d < D; ++d ) // for each dimension
        {
            evaluation[d].resize( rstPre[d].size( ) * voxelRst[d].size( ), 0.0 );

            polynomial::lagrange(   lagOrderPre, // order of the lagrange polynomial
                                    voxelRst[d].size( ), // Number of evaluation points
                                    rstPre[d].data( ), // Element integration points (Stützstellen)
                                    voxelRst[d].data( ), // Auswertungstellen
                                    evaluation[d].data( ) );
        }

        cache.lagrangeEvalPerLevel[0] = evaluation;


    } // else


    return cache;

}

template<size_t D>
size_t voxelMomentFittingQuadrature<D>::partition( const MeshMapping<D>& mapping,
                                                   QuadratureCache<D>& anyCache ) const
{
    auto& cache = utilities::cast<Cache>( anyCache );

    cache.mapping = &mapping;
    cache.icell = mapping.icell;
    cache.nvoxels = nvoxels_( mapping.icell );

    return 1;
}



template<size_t D>
bool voxelMomentFittingQuadrature<D>::distribute( size_t partition,
                                                  std::array<size_t, D> orders,
                                                  CoordinateGrid<D>& rst, // local partition coordinates
                                                  CoordinateList<D>& xyz, // mapped real coordinates
                                                  std::vector<double>& weights,
                                                  QuadratureCache<D>& anyCache ) const
{
    // Hand over material function
    const spatial::ScalarFunction<D>& E = alpha_;
    auto& data = utilities::cast<Cache>( anyCache ); // unpacking the data in the cache

    if( data.mapping->type == CellType::NCube ) // checking type of element
    {
        auto level = 0;
        if ( grid_ )
        {
            // Get current refinement level
            level = grid_->refinementLevel( grid_->fullIndex( data.icell ) );
        }

        // MOMENT FITTING PARAMETERS
        auto degree = polynomialDegree_; // DEGREE COULD BE RETRIEVED FROM ORDER OF INTEGRAND
        std::array<size_t, D> n_mf;
        std::fill( n_mf.begin(  ), n_mf.end( ), 2 * degree + 1 );
        //std::cout << n_mf[0] << std::endl;

        auto lag_order = ( 2 * degree + 1 ) - 1;

        // QUADRATURE ORDER FOR VOXEL WISE INTEGRATION OF LAGRANGE
        std::array<size_t, D> nvoxels = data.nvoxels;

        // Get root cell ip and weights, reset weights and get ips
        tensorProductQuadrature( n_mf, rst, weights, data.quadrature );
        std::fill(weights.begin(  ), weights.end(), 1.0);
        // orders is the number of respective IP points for exact integration
        mapQuadraturePointGrid( *data.mapping, rst, xyz, weights );


        // assign empty vector for storing the obtained moments
        auto moments = std::vector( weights.size( ), 0.0 );

        // Integration order on individual voxel
        auto voxelOrders = array::makeSizes<D>( static_cast<size_t>( std::ceil( ( lag_order + 1.0 ) / 2.0 ) ) );
        // Area scaling of individual voxel
        auto voxelDetJ = array::product( array::divide( 1.0, array::convert<double>( data.nvoxels ) ) );



        // get voxel coordinates and weights
        CoordinateGrid<D> voxelRst = data.voxelRstPerLevel[level];
        std::array<std::vector<double>, D> voxelWeights = data.voxelWeightsPerLevel[level];


        moments.resize( static_cast<size_t>( std::pow( lag_order + 1 , D ) ), 0.0 );

        std::array<size_t, D> nPts;
        size_t nPtsTotal = 1;
        for (size_t d = 0; d < D; ++d)
        {
            nPts[d] = voxelRst[d].size();
            nPtsTotal *= nPts[d];
        }

        auto evalLag = data.lagrangeEvalPerLevel[level];

        if ( !scaling_ ) // Checking if the split into tensor product is viable
        {
            // Over all voxel integration points
            // // Splitted tensor product integration
            std::array<std::vector<double>, D> moments1D;

            for (size_t d = 0; d < D; ++d)
            {
                const size_t nV = data.nvoxels[d];
                moments1D[d].resize(( lag_order + 1 ) * nV, 0.0);

                const double voxel_size = 2.0 / static_cast<double>( nV );

                for (size_t v = 0; v < nV; ++v)
                {
                    for (size_t ip = 0; ip < voxelRst[d].size(); ++ip)
                    {
                        const double w = voxelWeights[d][ip];

                        for (size_t lag = 0; lag < lag_order + 1; ++lag)
                        {
                            moments1D[d][lag * nV + v] +=
                                evalLag[d][lag * voxelRst[d].size() + ip]
                                * w * voxel_size;
                        }
                    }
                }
            }


            std::fill(moments.begin( ), moments.end( ), 0.0);

            nd::executeWithIndex(data.nvoxels,
            [&](std::array<size_t, D> voxel_idx, size_t)
            {
                std::array<double, D> voxel_center;

                for (size_t d = 0; d < D; ++d)
                {
                    double step = 2.0 / static_cast<double>( data.nvoxels[d] );
                    voxel_center[d] = ( voxel_idx[d] + 0.5 ) * step - 1;
                }

                auto [xyz, detJ] = map::withDetJ( *data.mapping, voxel_center );
                const double weight = 1 / std::pow(2, D) * voxelDetJ * E( xyz ) * detJ;

                std::array<size_t, D> max = array::makeSizes<D>(lag_order + 1);

                nd::executeWithIndex( max,
                [&](std::array<size_t, D> lmn, size_t index )
                {
                    double value = weight;

                    for (size_t d = 0; d < D; ++d)
                    {
                        value *= moments1D[d][lmn[d] * data.nvoxels[d] + voxel_idx[d]];
                    }
                    moments[index] += value;
                });
            });

        }

        else // run over all voxel integration points
        {
            const auto& scaling = scaling_.value(  );

            // Over all voxel integration points
            nd::executeWithIndex( nPts, [&](std::array<size_t, D> ijk, size_t index1 )
            {
                        std::array<double, D> rsm;
                        auto weight = 1.0;

                        for ( size_t d = 0; d < D; ++d )
                        {
                            rsm[d] = voxelRst[d][ijk[d]];
                            // assembled tensor product weight
                            weight *= voxelWeights[d][ijk[d]];
                        }

                        auto [xyz, detJ] = map::withDetJ( *data.mapping, rsm );
                        MLHP_CHECK( detJ > 0, "Jacobian is not positive." );
                        auto w_E = E( xyz );
                        // degradation function evaluated at voxel IP global coords: g ( s( xyz ) )
                        auto g_s = ( scaling( xyz ) == 0 ) ? 1e-9 : scaling( xyz ); // for conditioning of the matrix


                       // Over all lagrange basis functions
                        std::array<size_t, D> max = array::makeSizes<D>( lag_order + 1);
                        nd::executeWithIndex(max, [&]( std::array<size_t, D> lmn, size_t index2 )
                        {
                            // voxel young's modulus * phasefield value at voxel IP
                            double value = w_E * g_s * weight * voxelDetJ * detJ;
                            for ( size_t d = 0; d < D; ++d )
                            {
                                value *= evalLag[d][lmn[d] * voxelRst[d].size( ) + ijk[d]];
                            }

                            moments[index2] += value;

                        } ); // loop lagrange basis functions
            } ); // loop voxel integration points

        }


        weights = moments;
    }

    return true;
}
