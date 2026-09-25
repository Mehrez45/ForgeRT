#include <cuda_runtime.h>
#include <new>
#include <stdexcept>
#include <string>
#include <cstddef>
#include <type_traits>

template <typename T>
concept CudaCompatible = std::is_trivially_copyable_v<T>;

template <CudaCompatible T>
class CudaBuffer{
private:
    std::size_t count;
    T* ptr_d = nullptr;


public:

    explicit CudaBuffer(std::size_t count) : count(count) {
        if(count > 0){
            cudaError_t err = cudaMalloc(&ptr_d, sizeof(T) * count);
            if(err == cudaErrorMemoryAllocation){
                throw std::bad_alloc{};
            }
            if (err != cudaSuccess) {
                throw std::runtime_error(
                std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
            }
        }
        
    };

    CudaBuffer(CudaBuffer&& other) noexcept
        : count(other.count), ptr_d(other.ptr_d) {
        other.count = 0;
        other.ptr_d = nullptr;
    }

    CudaBuffer(const CudaBuffer& other) = delete;
    CudaBuffer& operator=(const CudaBuffer& other) = delete;

    CudaBuffer& operator=(CudaBuffer&& other) noexcept{
        if(this != &other){
            if(this->ptr_d != nullptr){
                cudaFree(ptr_d);
            }

            this->count = other.count;             
            this->ptr_d = other.ptr_d;             
        
            other.count = 0;              
            other.ptr_d = nullptr;
        }

        return *this;
    }

    ~CudaBuffer(){
        if (ptr_d != nullptr){
            cudaFree(ptr_d);
        }
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return count;
    }

    [[nodiscard]] T* data() noexcept {
        return ptr_d;
    }

    [[nodiscard]] const T* data() const noexcept {
        return ptr_d;
    }

    void copyFromHost(const T* sourcePtr, std::size_t elementCount) {
        if (elementCount == 0){
            return;
        } else if (elementCount > this->count) {
            throw std::out_of_range("Invalid argument - Element count exceeds buffer size");
        } else if (sourcePtr == nullptr) {
            throw std::invalid_argument("Invalid argument - Source pointer must not be null");
        }

        cudaError_t err = cudaMemcpy(this->ptr_d, sourcePtr, sizeof(T) * elementCount, cudaMemcpyHostToDevice);

        if (err != cudaSuccess){
            throw std::runtime_error(
                std::string("Cuda memory copy failed: ") + cudaGetErrorString(err));
        }

    }

    void copyToHost(T* destinationPtr, std::size_t resultCount){
        if (resultCount == 0){
            return;   
        } else if (destinationPtr == nullptr) {
            throw std::invalid_argument("The destination poiner must not be null");
        } else if (resultCount > count){
            throw std::out_of_range("Invalid argument - Result count exceeds buffer size");
        }

        cudaError_t err = cudaMemcpy(destinationPtr, this->ptr_d, sizeof(T) * resultCount, cudaMemcpyDeviceToHost);

        if (err != cudaSuccess){
            throw std::runtime_error(
                std::string("Cuda memory copy failed: ") + cudaGetErrorString(err));
        }

    }

};
