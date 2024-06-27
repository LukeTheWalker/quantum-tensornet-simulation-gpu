cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CuContraction LANGUAGES CXX)

# verbose makefile
set(CMAKE_VERBOSE_MAKEFILE on)

# make O3 the default
set(CMAKE_BUILD_TYPE "Release")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Option to enable/disable CUDA
option(ENABLE_CUDA "Enable CUDA support" OFF)

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED True)
    add_definitions(-DENABLE_CUDA)

    # Find CUDA package
    find_package(CUDA REQUIRED)
    
    # Include cuBLAS
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_LIBRARIES cublas)
endif()

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the source files
file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

if (ENABLE_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES
        ${PROJECT_SOURCE_DIR}/src/*.cu
    )
    list(APPEND CPP_SOURCES ${CUDA_SOURCES})
endif()

# Add the executable
add_executable(CuContraction ${CPP_SOURCES})

# Link against SQLite3
find_package(SQLite3 REQUIRED)
target_link_libraries(CuContraction PRIVATE SQLite::SQLite3)

if (ENABLE_CUDA)
    # Set target properties for CUDA
    set_target_properties(CuContraction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(CuContraction PROPERTIES CUDA_ARCHITECTURES 80)
    
    # Link against CUDA libraries including cuBLAS
    target_link_libraries(CuContraction PRIVATE ${CUDA_LIBRARIES})
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(CuContraction PRIVATE OpenMP::OpenMP_CXX)
endif()