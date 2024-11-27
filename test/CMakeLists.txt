# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(mnist-cppTests LANGUAGES CXX)

# ---- Tests ----

add_executable(mnist-cpp_test source/mnist-cpp_test.cpp)
target_link_libraries(mnist-cpp_test PRIVATE mnist-cpp_lib)
target_compile_features(mnist-cpp_test PRIVATE cxx_std_17)

add_test(NAME mnist-cpp_test COMMAND mnist-cpp_test)

# ---- End-of-file commands ----

add_folders(Test)
