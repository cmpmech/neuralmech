cmake_minimum_required(VERSION 3.18)
project(mlp_sine CXX)

find_package(Torch REQUIRED)
add_executable(mlp_sine mlp_sine.cpp)
target_link_libraries(mlp_sine "${TORCH_LIBRARIES}")
set_property(TARGET mlp_sine PROPERTY CXX_STANDARD 17)
