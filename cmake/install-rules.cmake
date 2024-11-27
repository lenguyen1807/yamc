install(
    TARGETS mnist-cpp_exe
    RUNTIME COMPONENT mnist-cpp_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
