cmake_minimum_required(VERSION 3.20)

project(MNIST)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} src/main.cpp 
                               src/matrix.cpp
                               include/matrix.h
                               src/image.cpp
                               include/image.h
                               src/nn.cpp
                               include/nn.h
                               include/activation.h)

add_compile_definitions(DATA_PATH="${PROJECT_SOURCE_DIR}/data/")