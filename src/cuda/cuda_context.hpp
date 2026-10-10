#pragma once
#include <string>
#include <cuda_runtime.h>

class CudaContext{
private:
    int deviceId;
    cudaDeviceProp properties_{};

public:
    explicit CudaContext(int deviceId = 0);

    [[nodiscard]] int device_id() const noexcept;
    [[nodiscard]] const cudaDeviceProp& properties() const noexcept;
    [[nodiscard]] std::string info_string() const;
    [[nodiscard]] static int device_count();

};
