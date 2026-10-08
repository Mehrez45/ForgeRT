#pragma once

#include <cuda_runtime.h>

class GpuTimer {
private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool started = false;
    bool stopped = false;

public:
    GpuTimer();
    ~GpuTimer() noexcept;

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start();
    void stop();
    float getDuration();
};