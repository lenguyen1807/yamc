cmake_minimum_required(VERSION 3.16)
project(mnist_cpp)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Thanks Claude for helping me fix this problem
if(APPLE)
    set(LLVM_PATH "/opt/homebrew/opt/llvm")
    
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

add_compile_definitions(DATA_SOURCE_DIR="${CMAKE_SOURCE_DIR}/data/")
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

file(GLOB_RECURSE SOURCES 
    "${SOURCE_DIR}/*.cpp"
)

add_library(
    mnist-cpp_lib OBJECT
    ${SOURCES}
)

# Add include directories
target_include_directories(mnist-cpp_lib ${warning_guard}
    PUBLIC
    ${INCLUDE_DIR}
    ${OpenMP_CXX_INCLUDE_DIRS}
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
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

# Include Catch2 for testing
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)

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

    target_link_libraries( 
        mnist-cpp_lib
        PUBLIC
        OpenMP::OpenMP_CXX 
        ${OpenCV_LIBS}
        ${ACCELERATE_LIBRARY} 
        ${FOUNDATION_LIBRARY} 
        ${METAL_LIBRARY} 
        ${QUARTZCORE_LIBRARY})
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
endif()

# Create executable with all source files
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE mnist-cpp_lib)

if(APPLE)
    SET(CMAKE_CXX_FLAGS "-O2 -ftree-vectorize -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -DACCELERATE_LAPACK_ILP64 -DACCELERATE_NEW_LAPACK")
else()
    SET(CMAKE_CXX_FLAGS "-O2 -ftree-vectorize")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(tests)