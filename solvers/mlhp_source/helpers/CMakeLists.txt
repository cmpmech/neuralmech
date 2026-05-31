# NeuralMech extension to mlhp. License: See mlhp/LICENSE
#
# Builds the pymlhphelpers python module with custom physics integrands. Lives
# outside the pinned mlhp submodule and links against the mlhpcore library and
# the pybind11 already pulled in by the mlhp subdirectory.

if( ${MLHP_PYTHON} )

    pybind11_add_module( pymlhphelpers integrand.cpp j2plasticity.cpp python/bindings.cpp )

    target_link_libraries( pymlhphelpers PRIVATE mlhpcore mlhp_private_compile_flags )

    # Source root resolves "src/python/pymlhpcore.hpp"; current dir resolves "integrand.hpp"
    target_include_directories( pymlhphelpers PRIVATE ${mlhp_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} )

    set_target_properties( pymlhphelpers PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${MLHP_BUILD_BINARY_DIR} )

    # Copy python wrapper next to the compiled module
    file( COPY python/mlhphelpers.py DESTINATION ${MLHP_BUILD_BINARY_DIR} )

endif( ${MLHP_PYTHON} )
