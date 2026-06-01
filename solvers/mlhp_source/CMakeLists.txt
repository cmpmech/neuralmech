cmake_minimum_required(VERSION 3.12)

project(mlhp_source LANGUAGES CXX)

# We expose a SINGLE python module (pymlhpcore) that bundles mlhp's bindings with the
# NeuralMech helper extensions (built in helpers/).  Keep mlhp's own python module off
# so it does not also create a pymlhpcore target, then add pybind11 ourselves and let
# helpers/ build the combined module.  Nothing in the mlhp submodule is changed.
option(MLHP_BUILD_PYTHON "Build the combined pymlhpcore python module." ON)
set(MLHP_PYTHON OFF CACHE BOOL "mlhp builds its own python module" FORCE)

add_subdirectory(mlhp)

if(MLHP_BUILD_PYTHON)
    add_subdirectory(mlhp/external/pybind11)
endif()

add_subdirectory(helpers)
