#include <algorithm>
#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

sycl::platform findMultiDevicePlatform() {
  auto platforms = sycl::platform::get_platforms();
  auto platform_ptr = std::find_if(platforms.begin(), platforms.end(),
                                   [](const sycl::platform& sycl_platform) {
                                     auto devices = sycl_platform.get_devices();
                                     if (1 < devices.size()) {
                                       return true;
                                     } else {
                                       return false;
                                     }
                                   });
  if (platforms.end() == platform_ptr) {
    std::cerr << "A multi-device platform is not available.\n";
    std::exit(EXIT_FAILURE);
  }
  return *platform_ptr;
}

int main() {
  sycl::platform sycl_platform = findMultiDevicePlatform();
  auto sycl_devices = sycl::device::get_devices();

  // Create a *single context* containing the first two devices

  // Create one queue on each of the first two devices

  // Create a host vector
  const size_t vector_length = 100;
  std::vector<float> host_vector_1(vector_length, 1.0);

  // Allocate a vector on each of the the first two devices

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
