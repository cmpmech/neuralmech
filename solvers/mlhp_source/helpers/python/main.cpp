// NeuralMech extension to mlhp. License: See mlhp/LICENSE
//
// Combined pybind module entry.  Produces a single `pymlhpcore` extension containing
// mlhp's own bindings plus the NeuralMech helper extensions, so the stock mlhp.py
// wrapper ("from pymlhpcore import *") and therefore `import mlhp` expose everything
// through one module / one import.
//
// The four mlhp::bindings::bind* functions are declared here exactly as mlhp's own
// src/python/main.cpp declares them (external-linkage free functions defined across
// its binding translation units).  We compile those translation units except main.cpp
// into this module and supply our own PYBIND11_MODULE, so the pinned mlhp submodule
// source is never modified.

#include "pybind11/pybind11.h"

namespace mlhp::bindings
{
void bindSpatial( pybind11::module& m );
void bindDiscretization( pybind11::module& m );
void bindAssembly( pybind11::module& m );
void bindPostprocessing( pybind11::module& m );
} // namespace mlhp::bindings

namespace mlhp::helpers
{
void bindHelpers( pybind11::module& m );
} // namespace mlhp::helpers

PYBIND11_MODULE( pymlhpcore, m )
{
    m.doc( ) = "Multi-level hp discretization kernel (with NeuralMech extensions).";

    mlhp::bindings::bindSpatial( m );
    mlhp::bindings::bindDiscretization( m );
    mlhp::bindings::bindAssembly( m );
    mlhp::bindings::bindPostprocessing( m );
    mlhp::helpers::bindHelpers( m );
}
