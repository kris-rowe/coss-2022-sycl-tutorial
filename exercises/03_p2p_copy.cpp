#include <algorithm>
#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

int main() {
  auto sycl_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (2 > sycl_devices.size()) {
    std::cout << "Only one GPU is available.\n";
    return EXIT_FAILURE;
  }

  // Select the first two devices

  // Create one context per device

  // Create one queue on each device

  // Create a host vector
  const size_t vector_length = 100;
  std::vector<float> host_vector_1(vector_length,1.0);

  // Allocate a vector the each of the devices

  // Copy values from the host to the first device
  
  // Copy values from the first device to the second

  // Copy values from the second device to the host
  std::vector<float> host_vector_2(vector_length);

  // Verify the results.
  for (const auto& x : host_vector_2) {
    // Don't check for equality of floating-point values in production code!
    if (1.0 != x) {
      std::cout << "Verification failed!\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  // Free device memory
  return EXIT_SUCCESS;
}
