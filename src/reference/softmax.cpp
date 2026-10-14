#include "softmax.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>


namespace forgeRT::reference {

    std::vector<float> softmax(const std::vector<float>& input) {
        if (input.empty()) throw std::invalid_argument("The input vector musn't be empty");

        std::vector<float> result(input.size(), 0);
        float max = input.front();

        for (float num : input) {
            max = std::max(max, num);
        }

        double denominator = 0.0;

        for (std::size_t i = 0; i < input.size(); ++i) {
            result[i] = std::exp(input[i] - max);
            denominator += result[i];
        }

        for (float& val : result) {
            val = static_cast<float>(val / denominator);
        }

        return result;
    }
}

