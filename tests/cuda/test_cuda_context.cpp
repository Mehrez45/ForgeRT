#include <cuda_context.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

bool validDevice() {
    try {
        CudaContext cudaContext(0);
        if(cudaContext.device_id() != 0) return false;

        int activeDevice = -1;
        cudaError_t status = cudaGetDevice(&activeDevice);

        if (status != cudaSuccess || activeDevice != 0) return false; 

        if (cudaContext.properties().name[0] == '\0') return false;

        std::cout << cudaContext.info_string();

    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return false;
    }

    return true; 
}

bool invalidDeviceNegativeId() {
    try {
        CudaContext cudaContext(-1);
    } catch (const std::invalid_argument& err) {
        return true;
    }

    std::cerr << "Expected std::invalid_argument was not thrown\n";
    return false;
}

bool invalidDeviceId() {
    try {
        int count = CudaContext::device_count();
        CudaContext cudaContext(count);
    } catch (const std::invalid_argument& err) {
        return true;
    }

    std::cerr << "Expected std::invalid_argument was not thrown\n";
    return false;
}

int main(){
    bool success = true;

    if (!validDevice()) success = false;
    if (!invalidDeviceNegativeId()) success = false;
    if (!invalidDeviceId()) success = false;

    if (success) {
        std::cout << "All CUDA context tests passed\n";
    }

    return success ? 0 : 1;
   
}