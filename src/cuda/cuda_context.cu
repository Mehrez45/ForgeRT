#include "cuda_context.hpp"
#include <stdexcept>
#include <sstream>

CudaContext::CudaContext(int deviceId) : deviceId(deviceId) {
    if (deviceId < 0) throw std::invalid_argument("Invalid CUDA device ID");

    int count = device_count();

    if (count == 0) throw std::runtime_error("No CUDA devices found");
    if (deviceId < 0 || deviceId >= count) throw std::invalid_argument("Invalid CUDA device ID");

    cudaError_t status = cudaSetDevice(deviceId);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(status));
    }

    status = cudaGetDeviceProperties(&properties_, deviceId);
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("cudaGetDeviceProperties failed: ")
        + cudaGetErrorString(status));
    }
    
}

int CudaContext::device_id() const noexcept {
    return deviceId;
}

const cudaDeviceProp& CudaContext::properties() const noexcept {
    return properties_;
}

std::string CudaContext::info_string() const {
    std::ostringstream oss;

    oss << "GPU: " << properties_.name << '\n';
    oss << "Compute Capability: "
        << properties_.major << "."
        << properties_.minor << '\n';
    oss << "Global Memory: "
        << properties_.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
        << " GiB\n";
    oss << "SM Count: "
        << properties_.multiProcessorCount << '\n';
    oss << "Max Threads Per Block: "
        << properties_.maxThreadsPerBlock << '\n';

    return oss.str();
}

int CudaContext::device_count() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);

    if (status != cudaSuccess){
        throw std::runtime_error(std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(status));
    }

    return count;
}
