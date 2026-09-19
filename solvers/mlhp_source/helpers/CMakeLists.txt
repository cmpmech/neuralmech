# NeuralMech extension to mlhp. License: See mlhp/LICENSE
#
# Builds ONE python module, pymlhpcore, that bundles mlhp's own pybind bindings with
# the custom NeuralMech integrands.  We compile mlhp's binding sources (minus its
# module entry main.cpp) together with the helper sources and our own combined entry
# (python/main.cpp).  The pinned mlhp submodule is never modified; the top-level
# CMakeLists forces mlhp's own MLHP_PYTHON off so it does not also create pymlhpcore.

if( ${MLHP_BUILD_PYTHON} )

    # mlhp's binding source list (sets MLHP_PYTHON_BINDING_SOURCES and
    # MLHP_PYTHON_PACKAGE_SOURCES), minus its module entry
    include( ${mlhp_SOURCE_DIR}/src/python/files.cmake )
    list( REMOVE_ITEM MLHP_PYTHON_BINDING_SOURCES main.cpp )
    list( TRANSFORM MLHP_PYTHON_BINDING_SOURCES PREPEND ${mlhp_SOURCE_DIR}/src/python/ )
    list( TRANSFORM MLHP_PYTHON_PACKAGE_SOURCES PREPEND ${mlhp_SOURCE_DIR}/src/python/mlhp/ )

    pybind11_add_module( pymlhpcore
        ${MLHP_PYTHON_BINDING_SOURCES}
        integrand.cpp
        python/bindings.cpp
        python/main.cpp )

    target_link_libraries( pymlhpcore PRIVATE mlhpcore mlhp_private_compile_flags )

    # "mlhp/python/pymlhpcore.hpp" comes from mlhpcore's public include dir; current dir
    # resolves "integrand.hpp"
    target_include_directories( pymlhpcore PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} )

    # Upstream package layout: mlhp/__init__.py does "from ._core import *", so the module
    # file is named _core and sits inside bin/mlhp/ next to the python sources and the
    # shared core library (found via $ORIGIN). `import mlhp` then exposes the whole API,
    # helpers included, through one package.
    set( MLHP_PACKAGE_DIR ${MLHP_BUILD_BINARY_DIR}/mlhp )

    set_target_properties( pymlhpcore PROPERTIES OUTPUT_NAME _core
                           LIBRARY_OUTPUT_DIRECTORY ${MLHP_PACKAGE_DIR}
                           BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN" )

    file( COPY ${MLHP_PYTHON_PACKAGE_SOURCES} DESTINATION ${MLHP_PACKAGE_DIR} )

    if( BUILD_SHARED_LIBS )
        add_custom_command( TARGET pymlhpcore POST_BUILD COMMAND ${CMAKE_COMMAND} -E
            copy_if_different $<TARGET_FILE:mlhpcore> ${MLHP_PACKAGE_DIR} )
    endif( BUILD_SHARED_LIBS )

endif( ${MLHP_BUILD_PYTHON} )
