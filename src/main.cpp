#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>
#include <cmath>
#include <chrono>
int main()
{
    constexpr auto n = 1000000000;
    auto numbers = std::vector<float>(n);
    std::iota(numbers.begin(), numbers.end(), 0);
    auto start = std::chrono::high_resolution_clock::now();
    std::transform(std::execution::par_unseq, numbers.begin(), numbers.end(), numbers.begin(),
                   [](auto a)
                   { return std::sqrt(a); });
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Took: [" << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "] ms" << std::endl;
}
