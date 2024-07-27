cmake_minimum_required(VERSION 3.20)

project(MNIST)

# download catch2 test
include(FetchContent)
Fetchcontent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0 # or a later release
)
FetchContent_MakeAvailable(Catch2)

# set C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# enable testing
enable_testing()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS "${PROJECT_SOURCE_DIR}/include/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/data/")

add_subdirectory("${PROJECT_SOURCE_DIR}/tests")