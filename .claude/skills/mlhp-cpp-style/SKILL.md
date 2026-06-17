---
name: mlhp-cpp-style
description: "C++ coding-style conventions for the mlhp finite element library. Use before writing or editing any non-trivial mlhp C++ so new code matches the existing codebase. Mandatory for work in mlhp, optional for downstream projects."
---

# mlhp - C++ code style

Only the conventions that are mlhp-specific or easy to get wrong; assume
ordinary modern-C++ good practice otherwise.

## New files

Every file starts with the license header, then a blank line:

```cpp
// This file is part of the mlhp project. License: See LICENSE
```

Headers use include guards, not `#pragma once`: `MLHP_CORE_<NAME>_HPP` in
the core library, `MLHP_BINDINGS_<NAME>_HPP` in the python bindings.

## Formatting

- Allman braces. **Every** control body is braced, opening brace on its
  own line. No single-statement omission and no `if( x ) return y;`
  (exception: a getter whose whole body is one line). 4-space indent, no tabs.
- Namespace bodies are **not** indented; nested namespaces each open on
  their own line and close with a `} // namespace x` comment.
- Spaces inside parentheses, including empty ones: `func( a, b )`,
  `if( cond )`, `begin( )`. **No** spaces inside brackets (`arr[i]`, not
  `arr[ i ]`). Spaces inside value-init braces: `std::array<T, N> { }`.
- One space around `=`; never pad to align columns (also for member
  initializer lists and structured bindings). The only alignment allowed
  is in data literals of test matrices.

```cpp
// correct                  // wrong - artificial padding
auto width = 4.0;           auto width  = 4.0;
auto height = 3.0;          auto height = 3.0;
auto area = width * height; auto area   = width * height;
```

```cpp
namespace mlhp::spatial
{

if( condition )
{
    auto result = std::array<double, D> { };

    for( size_t i = 0; i < n; ++i )
    {
        result[i] = process( i );
    }
}

} // namespace mlhp::spatial
```

## Naming

- Types: `PascalCase`. Functions and locals: `camelCase`. Private members:
  `camelCase` with a trailing underscore (`boundaryDofsMask_`).
- Namespaces: lowercase, per area (`mlhp`, `mlhp::spatial`, `mlhp::linalg`,
  `mlhp::array`, `mlhp::utilities`, `mlhp::implicit`, `mlhp::boundary`).
- Template parameters: `T`, `D` (spatial dim), `N` (array size),
  `G` (global dim), `L` (local dim), `M` (rows).

## Types and ownership

- `size_t` for all indices and sizes; `double` as the default float.
- `std::array` (fixed), `std::span` (non-owning view), `std::vector` (dynamic).
- Use `auto` for locals unless you see a good reason not to.
- Factory functions return `unique_ptr` or `shared_ptr`. Often, `shared_ptr` 
  is more convenient if the type is shared later.

## Error handling

Use the project macros - never `assert` or raw `throw`:

```cpp
MLHP_CHECK( n > 0, "Size must be positive." )    // always-on
MLHP_CHECK_DBG( i < size, "Index out of range." ) // debug-only
MLHP_EXPECTS( ptr != nullptr )                    // precondition (also MLHP_EXPECTS_DBG)
MLHP_THROW( "Descriptive message." )              // unconditional throw
MLHP_NOT_IMPLEMENTED                              // stub placeholder
```

## Attributes

`MLHP_PURE` (no side effects, result depends only on args) and
`MLHP_EXPORT` (exported from the shared library) go before the return type
(after `template<...>` if present); they combine with each other and with
`constexpr` (`MLHP_EXPORT MLHP_PURE double luDeterminant( ... )`).

## Comments

- Explain WHY, not WHAT. Keep comments short and concise. Don't comment
  what is obvious from the context (function/parameter names) or from
  general knowledge (no need to explain FEM-basics in an example driver).
- No decorative boxes or borders. Single-line header separators 
  (`// ========= Triangles =========`) are fine, used sparingly.
- `//!` doc comments in headers for doxygen, placed directly above the
  declaration with aligned parameter annotations; plain `//` in `.cpp`.

```cpp
//! Solve linear equation system M * x = b.
//! M : n x n matrix, overwritten by its LU decomposition
//! b : right hand side of size n
//! u : target solution vector of size n
MLHP_EXPORT void solve( std::span<double> M,
                        std::span<const double> b,
                        std::span<double> u );
```

## Lambdas

- One-liners stay on one line: `auto square = []( double x ) { return x * x; };`
- No hard rule for multi-line lambdas, but the usual habit: inline them in
  calls to the `nd::execute` family (and similar mlhp loop helpers), 
- If lambda is multi-line, keep the opening `{` on its own line at the call 
  statement's indentation, the body one level in, closing `} );`:

```cpp
nd::execute( array::makeSizes<D>( 2 ), [&]( std::array<size_t, D> corner )
{
    auto index = array::multiply( corner, ncells );
    appendConnectivity( nd::linearIndex( pointStrides, index ) );
} );
```

- For standard-library calls (`std::sort`, `std::transform`, ...) a longer
  lambda is instead declared as a named variable with the `{` on its own
  line at the `auto` column, then passed by name:

```cpp
auto predicate = [&]( size_t a, size_t b )
{
    return locationMap[a] < locationMap[b];
};

std::sort( indices.begin( ), indices.end( ), predicate );
```

- Prefer explicit captures (`[&foo, bar]`) in stored or long-lived
  lambdas; blanket `[&]`/`[=]` only for short, immediately-called ones.

## Constructor initializer lists

Colon at the end of the signature line, initializers indented one level
(several per line is fine):

```cpp
Foo( int x, int y ) :
    x { x },
    y { y }
{ }
```

## OpenMP

Hot loops use `#pragma omp parallel for`, kept consistent with the
patterns in `assembly.cpp`. `parallel.hpp` provides
`getMaxNumberOfThreads`, `getThreadNum`, `clampChunksize`.

## ASCII only

Comments and string literals - including pybind11 docstrings and
`MLHP_CHECK`/`MLHP_THROW` messages - must be plain ASCII. Spell out Greek
letters and use ASCII operators instead of Unicode math symbols or arrows:

```cpp
//! wrong:   Computes α·x + β·y; returns the Σ → normalized
//! correct: Computes alpha * x + beta * y; returns the sum -> normalized
```

## Output

Use `std::cout` / `std::cin` or `utilities::tic` / `toc`, not
`printf` / `scanf`. Prefer `std::endl` over `"\n"`; do not add a separate
`std::flush` after it (`std::endl` already flushes).

## Tests

Catch2, one file per feature named `*_test.cpp` (`tests/core/`,
`tests/system/`). `TEST_CASE( "<feature>_test" )`, grouped with `SECTION`.
Inside tests use Catch2 macros, not the `MLHP_*` ones: `CHECK` (continues
on failure) for most assertions, `REQUIRE` when later lines depend on it,
`CHECK_THROWS` for expected failures. Compare floating point with
`Approx( expected ).epsilon( ... ).margin( ... )`. Column-aligned numeric
literals are the one place padding is allowed (reference matrices).