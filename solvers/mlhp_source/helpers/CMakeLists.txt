# NeuralMech extension to mlhp. License: See mlhp/LICENSE
#
# Builds ONE python module, pymlhpcore, that bundles mlhp's own pybind bindings with
# the custom NeuralMech integrands.  We compile mlhp's binding sources (minus its
# module entry main.cpp) together with the helper sources and our own combined entry
# (python/main.cpp).  The pinned mlhp submodule is never modified; the top-level
# CMakeLists forces mlhp's own MLHP_PYTHON off so it does not also create pymlhpcore.

if( ${MLHP_BUILD_PYTHON} )

    # mlhp's binding source list (sets MLHP_PYTHON_BINDING_SOURCES), minus its entry
    include( ${mlhp_SOURCE_DIR}/src/python/files.cmake )
    list( REMOVE_ITEM MLHP_PYTHON_BINDING_SOURCES main.cpp )
    list( TRANSFORM MLHP_PYTHON_BINDING_SOURCES PREPEND ${mlhp_SOURCE_DIR}/src/python/ )

    pybind11_add_module( pymlhpcore
        ${MLHP_PYTHON_BINDING_SOURCES}
        integrand.cpp
        j2plasticity.cpp
        python/bindings.cpp
        python/main.cpp )

    target_link_libraries( pymlhpcore PRIVATE mlhpcore mlhp_private_compile_flags )

    # Source root resolves "src/python/pymlhpcore.hpp"; current dir resolves "integrand.hpp"
    target_include_directories( pymlhpcore PRIVATE ${mlhp_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

    set_target_properties( pymlhpcore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLHP_BUILD_BINARY_DIR} )

    # Upstream python wrapper: its "from pymlhpcore import *" now also pulls in the
    # helpers, so `import mlhp` exposes the whole API through one module.
    file( COPY ${mlhp_SOURCE_DIR}/src/python/mlhp.py DESTINATION ${MLHP_BUILD_BINARY_DIR} )

endif( ${MLHP_BUILD_PYTHON} )
