cmake_minimum_required(VERSION 3.16)
project(mnist_cpp)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Thanks Claude for helping me fix this problem
if(APPLE)
    set(LLVM_PATH "/opt/homebrew/opt/llvm")
    
    # NOTE: This will cause infinite loop in libcurl :>
    # # Set compiler to use LLVM Clang
    # set(CMAKE_C_COMPILER "${LLVM_PATH}/bin/clang")
    # set(CMAKE_CXX_COMPILER "${LLVM_PATH}/bin/clang++")
    
    # Explicitly set OpenMP flags
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LLVM_PATH}/lib/libomp.dylib")
    
    # Include directories for OpenMP
    set(OpenMP_C_INCLUDE_DIRS "${LLVM_PATH}/include")
    set(OpenMP_CXX_INCLUDE_DIRS "${LLVM_PATH}/include")
    
    # Set RPATH for runtime linking
    set(CMAKE_INSTALL_RPATH "${LLVM_PATH}/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE SOURCES 
    "${SOURCE_DIR}/*.cpp"
)

file(GLOB_RECURSE HEADERS 
    "${INCLUDE_DIR}/*.h"
)

# Create executable with all source files
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Add include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${INCLUDE_DIR}
        ${OpenMP_CXX_INCLUDE_DIRS}
)

# Find OpenMP package
find_package(OpenMP REQUIRED)
if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif()

# Find OpenCV package
find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenMP not found")
endif()

# Include libcurl to fetch data
# include(FetchContent)
# FetchContent_Declare(curl
# 	URL https://curl.se/download/curl-8.10.0.tar.gz
# 	DOWNLOAD_EXTRACT_TIMESTAMP true
# 	OVERRIDE_FIND_PACKAGE
# )
# FetchContent_MakeAvailable(curl)
# find_package(curl)

if(APPLE)
    # Link Apple framework
    find_library(ACCELERATE_LIBRARY Accelerate)
    if (NOT ACCELERATE_LIBRARY)
        message(FATAL_ERROR "Accelerate not found")
    endif()

    find_library(METAL_LIBRARY Metal)
    if (NOT METAL_LIBRARY)
        message(FATAL_ERROR "Metal not found")
    endif()

    find_library(FOUNDATION_LIBRARY Foundation)
    if (NOT FOUNDATION_LIBRARY)
        message(FATAL_ERROR "Foundation not found")
    endif()

    find_library(QUARTZCORE_LIBRARY QuartzCore)
    if (NOT QUARTZCORE_LIBRARY)
        message(FATAL_ERROR "QuartzCore not found")
    endif()

    SET(CMAKE_CXX_FLAGS "-O2 -DNDEBUG -std=c++17 -ftree-vectorize -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -DACCELERATE_LAPACK_ILP64 -DACCELERATE_NEW_LAPACK")
    target_link_libraries(${PROJECT_NAME} PRIVATE 
        OpenMP::OpenMP_CXX 
        ${OpenCV_LIBS}
        ${ACCELERATE_LIBRARY} 
        ${FOUNDATION_LIBRARY} 
        ${METAL_LIBRARY} 
        ${QUARTZCORE_LIBRARY})
else()
    SET(CMAKE_CXX_FLAGS "-O2 -DNDEBUG -std=c++17 -ftree-vectorize")
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)