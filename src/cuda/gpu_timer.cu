#include "gpu_timer.hpp"

#include <cuda_runtime.h>
#include <stdexcept>



GpuTimer::GpuTimer() : start_event(nullptr), stop_event(nullptr) {
    cudaError_t status = cudaEventCreate(&start_event);

    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }

    status = cudaEventCreate(&stop_event);

    if (status != cudaSuccess) {
        cudaEventDestroy(start_event);
        start_event = nullptr;
        throw std::runtime_error(cudaGetErrorString(status));
    }
}



GpuTimer::~GpuTimer() noexcept {
    if (start_event != nullptr) {
        cudaError_t status = cudaEventDestroy(start_event);

        if (status != cudaSuccess) {
            std::fprintf(stderr,
                         "Failed to destroy start event: %s\n",
                         cudaGetErrorString(status));
        }
    }

    if (stop_event != nullptr) {
        cudaError_t status = cudaEventDestroy(stop_event);

        if (status != cudaSuccess) {
            std::fprintf(stderr,
                         "Failed to destroy stop event: %s\n",
                         cudaGetErrorString(status));
        }
    }
}



void GpuTimer::start(){
    if (started && !stopped) throw std::logic_error("GpuTimer is already running");
    cudaError_t status = cudaEventRecord(start_event, 0);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));

    started = true;
    stopped = false;
}

void GpuTimer::stop(){
    if(!started) throw std::logic_error("Gputimer isn't running");
    cudaError_t status = cudaEventRecord(stop_event, 0);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));

    started = false;
    stopped = true;
}


float GpuTimer::getDuration() {
    if (!stopped) throw std::logic_error("GpuTimer has not been stopped");
    
    cudaError_t status = cudaEventSynchronize(stop_event);

    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    
    float elapsed_ms = 0.0f;
    status = cudaEventElapsedTime(
        &elapsed_ms,
        start_event,
        stop_event
    );

    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    
    return elapsed_ms;
}


