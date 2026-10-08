#pragma once

#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>
#include <functional>

#include "cuda_buffer.hpp"

template <typename T>
class Tensor {
private:
    std::vector<std::size_t> shape_;
    std::size_t numel_;
    CudaBuffer<T> buffer_;

public:
    explicit Tensor(std::vector<std::size_t> shape_)
        : shape_(std::move(shape_)), 
        numel_(std::accumulate(this->shape_.begin(), this->shape_.end(), std::size_t{1}, std::multiplies<>{})),
        buffer_(numel_)
    {}


    [[nodiscard]] size_t size() const noexcept {
        return buffer_.size();
    }

    [[nodiscard]] T* data() noexcept {
        return buffer_.data();
    }

    [[nodiscard]] const T* data() const noexcept {
        return buffer_.data();
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] const std::size_t numel() const noexcept {
        return numel_;
    }

    [[nodiscard]] std::size_t ndim() const noexcept {
        return shape_.size();
    }
};