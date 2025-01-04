set(TEST_INCLUDE test_helper.h)
set(TEST_SOURCES test_matrix.cpp test_activation.cpp)

add_executable(${PROJECT_NAME}_tests ${TEST_INCLUDE} ${TEST_SOURCES})

if(APPLE)
target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
        mnist-cpp_lib
        ${OpenCV_LIBS}
        ${ACCELERATE_LIBRARY} 
        ${FOUNDATION_LIBRARY} 
        ${METAL_LIBRARY} 
        ${QUARTZCORE_LIBRARY}
        Catch2::Catch2WithMain
        OpenMP::OpenMP_CXX
)
else()
target_link_libraries(${PROJECT_NAME}_tests
    PRIVATE
        mnist-cpp_lib
        ${OpenCV_LIBS}
        Catch2::Catch2WithMain
        OpenMP::OpenMP_CXX
)
endif()

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(${PROJECT_NAME}_tests)