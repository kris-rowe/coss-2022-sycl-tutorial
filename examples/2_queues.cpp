#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "CL/sycl.hpp"

// Check for (near) equality between two floating-point arrays
// Note: please be more rigorous than this in production code!
bool nearlyEqual(const std::vector<double>& lhs,const std::vector<double>& rhs) {
  for (std::size_t i{}; i < lhs.size(); ++i) {
    const double& expected = lhs[i];
    const double& actual = rhs[i];
    if (std::abs(expected - actual) >
        std::numeric_limits<double>::epsilon() * std::abs(expected + actual)) {
      return false;
    }
  }
  return true;
}

int main() {
  const std::size_t vector_length = 100;
  std::vector<double> host_original(vector_length);
  std::vector<double> host_copy(vector_length);

  // Create a vector of random values on the host
  std::random_device seed;
  std::mt19937_64 generator{seed()};
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  for (auto& x : host_original) {
    x = distribution(generator);
  }

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate a vector on the device
  double* device_vector =
      sycl::malloc_device<double>(vector_length, sycl_device, sycl_context);

  // Copy the vector of values from the host to the device
  sycl_queue.copy(host_original.data(), device_vector, vector_length);

  // Sychronize by waiting for all enqueued tasks to complete
  sycl_queue.wait();

  // Copy the values from the device back to the host
  sycl_queue.copy(device_vector, host_copy.data(), vector_length);
  sycl_queue.wait();

  if(!nearlyEqual(host_original,host_copy)) {
    std::cout << "Verification failed!\n";
    return EXIT_FAILURE;
  }  

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(device_vector, sycl_context);
  return 0;
}
