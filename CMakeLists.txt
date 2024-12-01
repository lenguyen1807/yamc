cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    mnist-cpp
    VERSION 0.1.0
    DESCRIPTION "A simple neural network to classify MNIST dataset in C++"
    HOMEPAGE_URL "https://lenguyen.vercel.app/projects/cpp-nn"
    LANGUAGES CXX
)

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

set(OPENBLAS_FOLDER ${PROJECT_SOURCE_DIR}/extern/OpenBLAS-0.3.28-x64-64)
if(NOT EXISTS "${OPENBLAS_FOLDER}")
    message(FATAL_ERROR "OpenBLAS folder not found: ${OPENBLAS_FOLDER}")
endif()


# ---- Declare library ----

add_library(
    mnist-cpp_lib OBJECT
    source/matrix.cpp
    source/utils.cpp
    source/linear.cpp
    source/mlp.cpp
    source/data.cpp
    source/optimizer.cpp
)

target_include_directories(
    mnist-cpp_lib ${warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${OPENBLAS_FOLDER}/include>" # openblas
    "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/source>"
)

target_link_libraries(
  mnist-cpp_lib
  "${OPENBLAS_FOLDER}/lib/libopenblas.lib"
)

# Ensure DLLs are copied to output directory
# For Debug builds
file(COPY 
    "${OPENBLAS_FOLDER}/bin/libopenblas.dll"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/Debug"
)

# For Release builds
file(COPY 
    "${OPENBLAS_FOLDER}/bin/libopenblas.dll"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/Release"
)

target_compile_features(mnist-cpp_lib PUBLIC cxx_std_17)
# add blas compile options
target_compile_definitions(mnist-cpp_lib PRIVATE
    USE_64BITINT
    DYNAMIC_ARCH
    DYNAMIC_OLDER
    NO_CBLAS
    NO_LAPACK
    NO_LAPACKE
    NO_AFFINITY
    USE_OPENMP
    GENERIC
    MAX_THREADS=64
)

# ---- Declare executable ----

add_executable(mnist-cpp_exe source/main.cpp)
add_executable(mnist-cpp::exe ALIAS mnist-cpp_exe)

set_property(TARGET mnist-cpp_exe PROPERTY OUTPUT_NAME mnist-cpp)

target_compile_features(mnist-cpp_exe PRIVATE cxx_std_17)

target_link_libraries(mnist-cpp_exe PRIVATE mnist-cpp_lib)

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# Path to data folder
add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/data/")

# ---- Developer mode ----

if(NOT mnist-cpp_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of mnist-cpp"
  )
endif()

include(cmake/dev-mode.cmake)
