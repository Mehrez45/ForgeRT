find_package(CUDAToolkit REQUIRED)

add_executable(test_cuda_context
    cuda/test_cuda_context.cpp
    "${PROJECT_SOURCE_DIR}/src/cuda/cuda_context.cu"
)

target_include_directories(test_cuda_context PRIVATE
    "${PROJECT_SOURCE_DIR}/src/cuda"
)

target_link_libraries(test_cuda_context PRIVATE
    CUDA::cudart
)