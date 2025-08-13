cmake_minimum_required(VERSION 3.18)
project(torchevent LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS_RELEASE "-fPIC -Wall -Wextra -O3 -U_FORTIFY_SOURCE")
set(CMAKE_CXX_FLAGS_RELEASE "-shared -O3 -Wunused-result -fopenmp")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

set(SOURCE_DIR "src/dpu")
include_directories(${SOURCE_DIR})
SET (LIB_FILES ${SOURCE_DIR}/core.cpp ${SOURCE_DIR}/transform.cpp ${SOURCE_DIR}/utils.cpp ${SOURCE_DIR}/variables.cpp)

# Combine static library for linked lib
ADD_LIBRARY (argo_lib_cpp STATIC ${LIB_FILES} )

find_package(pybind11 REQUIRED)

# Module Build and Move So file to PyPackage folder
pybind11_add_module(libargo ${SOURCE_DIR}/libargo.cpp)

target_link_libraries(libargo
  PRIVATE
    argo_lib_cpp
    pybind11::module
    /usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.so   # ← 절대경로로 직접 지정
)
set_property(TARGET libargo PROPERTY CXX_STANDARD 14)

set_target_properties(libargo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/torchevent
)
