---
name: mlhp-cpp-utilities
description: "Catalog of existing helper functions in the mlhp finite element library (array::, spatial::, linalg::, utilities:: namespaces). Consult before writing C++ in or against mlhp, to avoid reimplementing an existing helper."
---

# mlhp — utility and helper function reference

Helpers included transitively via most headers. Headers live under
`include/mlhp/core/`; implementation-only overloads and `detail::`
internals are omitted.

---

## `namespace mlhp::array` — `arrayfunctions.hpp`

Constexpr `std::array` utilities. All functions are non-mutating and
return a new array unless stated otherwise.

### Construction

| Function | Description |
|---|---|
| `array::make<N>( value )` | Fill all N elements with `value` |
| `array::makeSizes<N>( value )` | Shorthand `make<N, size_t>( value )` |
| `array::makeAndSet<N>( default, index, override )` | Fill with default, then set one entry |
| `array::range<T, N>( start=0, step=1 )` | Evenly-spaced sequence |

### Element access / reshape

| Function | Description |
|---|---|
| `array::slice( arr, normal )` | Drop element at `normal`, return `array<T, N-1>` |
| `array::sliceIfNotOne( arr, normal )` | `slice` only if `N > 1` |
| `array::insert( arr, index, value )` | Insert before `index`, return `array<T, N+1>` |
| `array::append( arr, value )` | Insert at end, return `array<T, N+1>` |
| `array::setEntry( arr, index, value )` | Return copy with one entry overridden |
| `array::peel( arr, index=N-1 )` | Returns `tuple{ slice(arr, index), arr[index] }` |
| `array::reverse( arr )` | Reverse element order |
| `array::duplicate<times>( arr )` | Return `array<array<T,N>, times>` |
| `array::midpoint( arr0, arr1 )` | Element-wise `std::midpoint` |
| `array::convert<TargetType>( arr )` | Cast each element |

### Element-wise arithmetic

All ops have three overloads: array x array, array x scalar, scalar x array.

| Function | Description |
|---|---|
| `array::add( a, b )` | `a + b` |
| `array::subtract( a, b )` | `a - b` |
| `array::multiply( a, b )` | `a * b` |
| `array::divide( a, b )` | `a / b` |
| `array::square( arr )` | `arr * arr` |
| `array::inverse( arr )` | `1 / arr` |
| `array::abs( arr )` | `std::abs` element-wise |

Infix `+`, `-`, `*`, `/` operators for `std::array<double, D>` also exist
(see the array-operators section below).

### Reduction

| Function | Description |
|---|---|
| `array::maxElement( arr )` | Maximum value |
| `array::minElement( arr )` | Minimum value |
| `array::maxArray( arr1, arr2 )` | Element-wise max (also scalar overload) |
| `array::minArray( arr1, arr2 )` | Element-wise min (also scalar overload) |
| `array::product( arr )` | Fold with `*` |
| `array::sum( arr )` | Fold with `+` |
| `array::accumulate( arr, op, initial )` | Generic fold |

### Containers of arrays

| Function | Description |
|---|---|
| `array::extract( vectors, ijk )` | `result[i] = vectors[i][ijk[i]]` |
| `array::elementSizes( containers )` | Array of `.size()` values |
| `array::resize( vectors, sizes )` | Resize each `std::vector` in an array |
| `array::resize0( vectors )` | Resize all to 0 |
| `array::to_string( arr )` | Format as `"(a, b, c)"` |

---

## `namespace mlhp::spatial` — `spatial.hpp`

### Vector math

| Function | Description |
|---|---|
| `spatial::dot( v1, v2 )` | Dot product (`array` or `span` overloads) |
| `spatial::norm( arr )` | Euclidean norm |
| `spatial::normSquared( arr )` | Squared norm |
| `spatial::normalize( v )` | Unit vector |
| `spatial::normalizeChecked( v, zero, tol )` | Returns zero vector if below tolerance |
| `spatial::distance( a, b )` | Euclidean distance |
| `spatial::distanceSquared( a, b )` | Squared distance |
| `spatial::invert( v )` | Element-wise `1 / v` |
| `spatial::cross( v1, v2 )` | 3-D cross product; 2-D returns scalar (z-component) |
| `spatial::interpolate( v1, v2, t )` | `t*v1 + (1-t)*v2` |
| `spatial::orthogonalize( u, v )` | Remove projection of `v` onto `u` |
| `spatial::projectionFactor( u, v )` | `dot(v,u)/dot(u,u)` |
| `spatial::standardBasisVector<N>( axis )` | Unit vector along `axis` |
| `spatial::findPlaneVectors( normal )` | Two orthonormal vectors spanning the plane |

### Bounding boxes

`BoundingBox<D>` is `std::array<std::array<double, D>, 2>` — `[min, max]`.

| Function | Description |
|---|---|
| `spatial::makeFullBoundingBox<D>()` | Init with `numeric_limits` (covers everything) |
| `spatial::makeEmptyBoundingBox<D>()` | Inverted limits (empty set) |
| `spatial::boundingBoxOr( b1, b2 )` | Union (smallest box covering both) |
| `spatial::boundingBoxOr( b, xyz )` | Extend box to include a point |
| `spatial::boundingBoxAnd( b1, b2 )` | Intersection |
| `spatial::extendBoundingBox( b, rel, abs )` | Inflate by relative + absolute margin |
| `spatial::boundingBoxAt( xyz, width )` | Centered box of given half-width |
| `spatial::insideBoundingBox( b, xyz )` | Point-in-box test |
| `spatial::boundingBoxIsValid( b )` | True if min <= max in all axes |
| `spatial::boundingBoxVolume( b )` | Product of side lengths |
| `spatial::boundingBox( coordinates )` | Compute tight bounding box of a point set |
| `spatial::boundingBoxIntersectsOther( b0, b1 )` | AABB overlap test |
| `spatial::boundingBoxIntersectsRay( b, origin, dir )` | Returns `optional<array<double,2>>` {t_enter, t_exit} |
| `spatial::boundingBoxIntersectsInvRay( b, origin, invDir )` | Same but with precomputed `1/dir` |

### Geometric primitives

| Function | Description |
|---|---|
| `spatial::multilinearShapeFunctions<D>( rst, diff )` | Bi/trilinear shape values or derivatives |
| `spatial::multilinearJacobian( corners, rst )` | Jacobian of multilinear map |
| `spatial::simplexShapeFunctions<D>( rst, diff )` | Simplex shape functions |
| `spatial::computeDeterminant( matrix )` | Det of Jacobian matrix |
| `spatial::concatenateJacobians( inner, outer )` | Chain rule: `outer * inner` |
| `spatial::mapPlaneNormal( jacobian, normal )` | Transform surface normal under deformation |
| `spatial::isDiagonal( matrix )` | Test if Jacobian is diagonal |

### Triangles and simplices

| Function | Description |
|---|---|
| `spatial::triangleCentroid( v0, v1, v2 )` | Centroid |
| `spatial::triangleNormal( v0, v1, v2 )` | Unit normal |
| `spatial::triangleCross( v0, v1, v2 )` | Unscaled normal (cross product of edges) |
| `spatial::triangleArea( v0, v1, v2 )` | Area (or span of overloads) |
| `spatial::tetrahedronVolume( v0, v1, v2, v3 )` | Volume |
| `spatial::triangleRayIntersection( v0, v1, v2, origin, dir )` | Returns `optional<double>` (t) |
| `spatial::simplexRayIntersection( vertices, origin, dir )` | Generic D=1/2/3 |
| `spatial::simplexMeasure( vertices )` | Length / area / volume |
| `spatial::simplexSubdivisionIndices<D>()` | Simplex sub-divisions of an n-cube |
| `spatial::simplexNormal( vertices )` | Normal to a simplex face |
| `spatial::clipPolygon( polygon, target, axis, pos, side )` | Sutherland-Hodgman clip |
| `spatial::clipTriangle( v0, v1, v2, axis, pos, side, target )` | Clip triangle to half-space |
| `spatial::clipLineSegment( v0, v1, bounds )` | Clip segment to AABB in-place |
| `spatial::segmentRayIntersection( v1, v2, origin, axis )` | 2-D segment x ray |
| `spatial::projectOntoLine( line0, line1, point )` | Returns `{point, t}` |
| `spatial::closestPointOnSegment( line0, line1, point )` | Returns `{point, t}` with t in [0,1] |

### Spatial functions and wrappers

`ScalarFunction<D>` is `std::function<double(array<double,D>)>`.
`VectorFunction<L, G>` is a type-erased wrapper around an L-in / G-out
spatial function; `G` can be `std::dynamic_extent`.

| Function / type | Description |
|---|---|
| `spatial::constantFunction<D>( value )` | Constant scalar |
| `spatial::constantFunction<I, O>( array )` | Constant vector |
| `spatial::VectorFunction<L, G>` | Construct from any compatible callable |
| `spatial::voxelFunction( data, nvoxels, lengths, origin )` | Piecewise-constant from voxels |
| `spatial::mask( func, implicit, alpha )` | Zero (or `alpha`*func) outside domain |
| `spatial::selectField( domains, default )` | Priority-ordered multi-domain field |
| `spatial::slice( func, axis, value )` | Fix one spatial coordinate |
| `spatial::sliceLast( func, value )` | Fix last coordinate |
| `spatial::expandDimension( func, pos )` | Ignore extra input dimensions |
| `spatial::extractComponent( vfunc, i )` | Scalar from vector field |
| `spatial::asVectorField( scalar )` | Wrap scalar as single-component vector |
| `spatial::centerNormalizedGaussBell( center, sigma )` | Bell with peak = scaling |
| `spatial::integralNormalizedGaussBell( center, sigma )` | Bell with integral = scaling |

### Homogeneous transformations

`HomogeneousTransformation<D>` supports method chaining
(`.translate().scale().rotate()`).

| Function | Description |
|---|---|
| `spatial::translate<D>( vector )` | Translation |
| `spatial::scale<D>( factors )` | Axis-aligned scaling |
| `spatial::rotate( phi )` | 2-D rotation |
| `spatial::rotate( normal, phi )` | 3-D rotation around axis |
| `spatial::concatenate( t1, t2, ... )` | Compose transforms |

### Coordinate generation

| Function | Description |
|---|---|
| `spatial::cartesianTickVectors( ncells, lengths, origin )` | 1-D tick vectors per axis |
| `spatial::tensorProduct( grid )` | Full coordinate list from tick vectors |
| `spatial::distributeSeedPoints( type, n, rst )` | Seed points inside a reference cell |
| `spatial::fibonacciSphere( n )` | ~Uniform points on the unit sphere |
| `spatial::findVoxel( nvoxels, lengths, origin, xyz )` | Index of voxel containing xyz |

---

## `namespace mlhp::linalg` — `dense.hpp`

All functions take `std::span<double>` views; allocate storage outside.
Matrices are row-major. All `*p*` vectors are size-`n` permutation
storage.

### All-in-one solvers

| Function | Description |
|---|---|
| `linalg::solve( M, p, b, u )` | Solve M*x = b; overwrites M with its LU |
| `linalg::invert( source, p, target )` | Invert n x n matrix |
| `linalg::det( M, p )` | Determinant (overwrites M) |

### LU decomposition (building blocks)

| Function | Description |
|---|---|
| `linalg::luFactor( M, p, eps )` | In-place LU with partial pivoting; returns swap count or -1 |
| `linalg::luSubstitute( LU, p, b, u )` | Forward/back substitution |
| `linalg::luDeterminant( LU, p, luResult )` | Det from existing factorisation |
| `linalg::luInvert( LU, p, I )` | Inversion from existing factorisation |

### Cholesky (symmetric positive definite, no pivot vector)

| Function | Description |
|---|---|
| `linalg::choleskyFactor( M, checkSymmetry=true )` | In-place factorisation; returns false if not SPD |
| `linalg::choleskySubstitute( L, b, u )` | Forward/back substitution |
| `linalg::choleskyDeterminant( L )` | Det from existing factorisation |
| `linalg::choleskyInvert( L, I )` | Inversion from existing factorisation |

### Decompositions and eigensolvers

| Function | Description |
|---|---|
| `linalg::qr( M, Q, R, sizes, reduce )` | QR decomposition (m x n, m >= n) |
| `linalg::hessenberg( A, tmp, V )` | Reduce to Hessenberg form in-place |
| `linalg::eigh( A, lambda, tmp )` | Symmetric eigenproblem; eigenvectors in rows of A |
| `linalg::eigh( A, B, lambda, tmp )` | Generalised symmetric eigenproblem A*v = lambda*B*v |
| `linalg::eig( A, real, imag, V )` | Non-symmetric eigenvalues; optional eigenvectors |
| `linalg::sorteig( lambda, eigenvectors, abs, asc )` | Sort eigendecomposition |
| `linalg::eigh2D( A )` | 2x2 symmetric eigenvalues (small, no allocation) |

### Matrix/vector products

| Function | Description |
|---|---|
| `linalg::mmproduct( left, right, target, size )` | Square n x n product |
| `linalg::mmproduct( left, right, target, M, K, N )` | General M x K x K x N |
| `linalg::mvproduct<M, N>( A, x, result )` | M x N matrix times N-vector (in-place) |
| `linalg::mvproduct<M, N>( A, x )` | Same, returns `array<double, M>` |

### Misc

| Function | Description |
|---|---|
| `linalg::norm( data )` | Euclidean norm of a vector |
| `linalg::normSquared( data )` | Squared norm |
| `linalg::identity( M, size )` | Fill n x n identity |
| `linalg::issymmetric( M, size, tol )` | Symmetry check |
| `linalg::givens( a, b )` | Givens rotation `[c, s]` |
| `linalg::givensr( a, b )` | Givens rotation `[c, s, r]` |
| `linalg::householderVector( x, v )` | Householder reflector (v, beta) |

---

## `namespace mlhp::utilities` — `utilities.hpp`

### Compile-time math

| Function | Description |
|---|---|
| `utilities::binaryPow<T>( exp )` | `2^exp` as type `T` |
| `utilities::integerPow( base, exp )` | `base^exp` (constexpr) |
| `utilities::factorial( n )` | n! (constexpr) |
| `utilities::divideCeil( a, b )` | Integer ceiling division |

### Scalar helpers

| Function | Description |
|---|---|
| `utilities::interpolate( x0, y0, x1, y1, x )` | Linear interpolation between two points |
| `utilities::mapToLocal0( begin, end, x )` | Map x to [0, 1] |
| `utilities::mapToLocal1( begin, end, x )` | Map x to [-1, 1] |
| `utilities::findInterval( positions, x )` | Binary search in sorted vector |
| `utilities::linspace( min, max, n )` | Equally-spaced `vector<double>` |

### Span helpers

| Function | Description |
|---|---|
| `utilities::spans( array_of_vecs )` | `array<span<T>, N>` from `array<vector<T>, N>` |
| `utilities::cspans( array_of_vecs )` | Const span version |
| `utilities::span( arr )` | `span<T, N>` from `array<T, N>` |
| `utilities::cspan( arr )` | `span<const T, N>` from `array<T, N>` or `span<T, N>` |

### Vector utilities

| Function | Description |
|---|---|
| `utilities::scaleVector( vec, value )` | In-place element-wise multiply |
| `utilities::addVectorsInplace( v1, v2 )` | `v1 += v2` element-wise |
| `utilities::convertVector<Target>( source )` | Cast element type |
| `utilities::linearizeVectors( vecs )` | Flatten `vector<vector<T>>` to `vector<T>` |
| `utilities::linearizedSpan( linearized, i )` | Span of element `i` in a linearized structure |
| `utilities::containerSizes( c... )` | Array of `.size()` values from variadic containers |
| `utilities::increaseSizes( n, c... )` | Extend each container by `n`, return old sizes |
| `utilities::vectorInternalMemory( v... )` | Total byte size of vector data |
| `utilities::resize0( v )` | Resize to 0 (overloaded for vector, array-of-vectors, variadic) |
| `utilities::clearMemory( v )` | Resize to 0 and release capacity |

### CSR / linearization helpers

| Function | Description |
|---|---|
| `utilities::allocateLinearizationIndices<I>( size )` | Allocate zero-filled index vector |
| `utilities::sumLinearizationIndices( indices )` | Prefix-sum in-place |
| `utilities::sumAndAllocateData( indices, value )` | Prefix-sum + allocate data vector |

### Timing and logging

```cpp
auto t = utilities::tic( "Assembling... " );   // print + capture time point
utilities::toc( t, "", " s.\n" );              // print elapsed time
double s = utilities::toc( t );               // just the elapsed seconds
```

### Lambdas

| Function | Description |
|---|---|
| `utilities::doNothing()` | Returns `[](auto&&...) noexcept {}` |
| `utilities::returnEmpty<T>()` | Returns `[](auto&&...) noexcept { return T{}; }` |
| `utilities::returnValue( obj )` | Returns constant-returning lambda |

### Misc

| Function | Description |
|---|---|
| `utilities::divideIntoChunks( size, nchunks, minChunk )` | Returns `[nchunks, minSize, firstSmall]` |
| `utilities::chunkRange( i, data )` | `[begin, end)` of chunk `i` |
| `utilities::copyShared( obj )` | Wrap in `shared_ptr` by copy |
| `utilities::moveShared( obj )` | Wrap in `shared_ptr` by move |
| `utilities::floatingPointEqual( b1, e1, b2, tol )` | Range comparison with tolerance |
| `utilities::ThreadLocalContainer<T>` | Per-thread data (indexed by `parallel::getThreadNum()`) |
| `utilities::Cache<Tag>` + `cast<T>( cache )` | Type-erased cache with optional debug type check |

### Macros (defined in `utilities.hpp`)

| Macro | Meaning |
|---|---|
| `MLHP_CHECK( expr, msg )` | Throw `std::runtime_error(msg)` if `expr` is false |
| `MLHP_CHECK_DBG( expr, msg )` | Same, compiled out without `MLHP_DEBUG_CHECKS` |
| `MLHP_EXPECTS( expr )` | Precondition check (same as `MLHP_CHECK` with generic message) |
| `MLHP_EXPECTS_DBG( expr )` | Debug-only precondition |
| `MLHP_THROW( msg )` | Unconditional throw |
| `MLHP_NOT_IMPLEMENTED` | Throw "not implemented" with function name |

---

## `namespace mlhp` — array operators (`spatial.hpp`)

`std::array<double, D>` has arithmetic operators defined in `namespace mlhp`:

```cpp
arr1 + arr2     arr1 - arr2     arr1 * arr2     arr1 / arr2
arr  + scalar   arr  - scalar   arr  * scalar   arr  / scalar
scalar + arr    scalar - arr    scalar * arr    scalar / arr
```

These are element-wise and cover the common coordinate arithmetic
(`xyz + offset * direction`, etc.) without going through
`array::add(...)`.

---

## Other headers worth knowing

- **`ndarray.hpp`** — multi-dimensional indexing: `nd::linearIndex( strides, ijk )`
  and `nd::unravel( index, limits )` (multi-index <-> flat, row-major),
  `nd::execute` / `nd::executeWithIndex` (loop over an index space),
  `nd::StaticArray`.
- **`memory.hpp`** — `AlignedDoubleVector` (SIMD-aligned storage used for
  element matrices/vectors in assembly).
- **`parallel.hpp`** — `parallel::getMaxNumberOfThreads()`,
  `parallel::getThreadNum()`, `parallel::clampChunksize()`.
- **`numeric.hpp`** — `newtonRaphson( evaluate, locationMaps, boundaryDofs, linearSolve, ... )`,
  `lineSearch( f, nsteps )`.