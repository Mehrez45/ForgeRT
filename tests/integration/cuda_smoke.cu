#include <cuda_runtime.h>
#include <iostream>

__global__ void sum(const int* a, const int* b, int* result) {
    const size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        *result = *a + *b;
    }
}

__host__ cudaError_t sum_h(int a, int b, int* result) {
    if (result == nullptr) {
        return cudaErrorInvalidValue;
    }

    int* result_d = nullptr;
    int* a_d = nullptr;
    int* b_d = nullptr;
    cudaError_t err = cudaSuccess;

    do {
        err = cudaMalloc(&result_d, sizeof(*result_d));
        if (err != cudaSuccess) break;

        err = cudaMalloc(&a_d, sizeof(*a_d));
        if (err != cudaSuccess) break;

        err = cudaMalloc(&b_d, sizeof(*b_d));
        if (err != cudaSuccess) break;

        err = cudaMemcpy(a_d, &a, sizeof(a), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) break;

        err = cudaMemcpy(b_d, &b, sizeof(b), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) break;

        sum<<<1, 1>>>(a_d, b_d, result_d);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) break;

        err = cudaMemcpy(result, result_d, sizeof(*result), cudaMemcpyDeviceToHost);
    } while (false);

    auto freeDeviceMemory = [&err](int* ptr) {
        if (ptr != nullptr) {
            const cudaError_t freeErr = cudaFree(ptr);
            if (err == cudaSuccess && freeErr != cudaSuccess) {
                err = freeErr;
            }
        }
    };

    freeDeviceMemory(result_d);
    freeDeviceMemory(a_d);
    freeDeviceMemory(b_d);
    return err;
}

int main() {
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);

    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << '\n';
        return 1;
    }
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    std::cout << "Number of CUDA Devices (GPUs): " << deviceCount << '\n';
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp properties{};
        const cudaError_t propertiesError = cudaGetDeviceProperties(&properties, i);
        if (propertiesError != cudaSuccess) {
            std::cerr << "Error querying GPU " << i << ": "
                      << cudaGetErrorString(propertiesError) << '\n';
            return 1;
        }

        std::cout << "GPU " << i << ": " << properties.name << '\n'
                  << "  Compute capability: " << properties.major << '.'
                  << properties.minor << '\n'
                  << "  Global memory: "
                  << properties.totalGlobalMem / (1024 * 1024) << " MiB\n"
                  << "  Max threads per SM: "
                  << properties.maxThreadsPerMultiProcessor << '\n'
                  << "  Max threads per block: "
                  << properties.maxThreadsPerBlock << '\n';
    }

    int result = 0;
    constexpr int expected = 27 + 65;
    status = sum_h(27, 65, &result);
    if (status != cudaSuccess) {
        std::cerr << "CUDA sum failed: " << cudaGetErrorString(status) << '\n';
        return 1;
    }

    if(result != expected) {
        std::cerr << "Incorrect CUDA result: expected "
              << expected << ", got " << result << '\n';
        return 1;
    }

    std::cout << "\n27 + 65 = " << result << '\n';
    return 0;
}
