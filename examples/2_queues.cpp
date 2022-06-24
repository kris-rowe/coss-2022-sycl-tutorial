#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

int main() {
  const std::size_t vector_length = 100;
  std::vector<double> host_original(vector_length,1.0);
  std::vector<double> host_copy(vector_length);

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate a vector on the device
  double* device_vector =
      sycl::malloc_device<double>(vector_length, sycl_device, sycl_context);

  // Copy the vector of values from the host to the device
  sycl_queue.copy(host_original.data(), device_vector, host_original.size());

  // Sychronize by waiting for all enqueued tasks to complete
  sycl_queue.wait();

  // Copy the values from the device back to the host
  sycl_queue.copy(device_vector, host_copy.data(), host_copy.size());
  sycl_queue.wait();

  // Verify the results.
  for(const auto& x : host_copy) {
    // Don't check for equality of floating-point values in production code!
    if(1.0 != x) {
      std::cout << "Verification failed!\n";
      return EXIT_FAILURE;
    }  
  }

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(device_vector, sycl_context);
  return 0;
}
