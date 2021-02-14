cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(classify_example_2 CXX)

list(APPEND CMAKE_PREFIX_PATH ~/Torch/libtorch)
find_package(Torch REQUIRED)

add_executable(classify_ex2 main.cxx)
add_compile_options(-Wall)
target_link_libraries(classify_ex2 "${TORCH_LIBRARIES}")
target_include_directories(classify_ex2 PUBLIC include)
set_property(TARGET classify_ex2 PROPERTY CXX_STANDARD 14)

