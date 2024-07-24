file(GLOB TESTS "${PROJECT_SOURCE_DIR}/tests/*.cpp")
add_executable(tests fibo_test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_BINARY_DIR}/_deps/catch2-src/extras") 
include(CTest)
include(Catch)
catch_discover_tests(tests)