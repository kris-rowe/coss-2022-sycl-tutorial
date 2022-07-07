#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
  const size_t vector_length = 2000;
  std::vector<float> a_host(vector_length);
  std::vector<float> b_host(vector_length, 1.0);
  std::vector<float> c_host(vector_length, 1.0);
  const float s = 1.0;

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate vectors on the device
  float* a =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);
  float* b =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);
  float* c =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);

  // Copy from the host to the device; synchronize.
  sycl_queue.copy(b_host.data(), b, b_host.size());
  sycl_queue.copy(c_host.data(), c, c_host.size());
  sycl_queue.wait();

  // Submit work to the queue using a kernel defined via lambdas; synchronize.
  sycl_queue.parallel_for({vector_length},
                          [=](sycl::id<1> i) { a[i] = b[i] + s * c[i]; });
  sycl_queue.wait();

  // Copy from the device to the host; synchronize.
  sycl_queue.copy(a, a_host.data(), a_host.size());
  sycl_queue.wait();

  // Verify the results.
  for (const auto& a_i : a_host) {
    // Don't check for equality of floating-point values in production code!
    if (2.0 != a_i) {
      std::cout << "Verification failed!\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(a, sycl_context);
  sycl::free(b, sycl_context);
  sycl::free(c, sycl_context);
  return EXIT_SUCCESS;
}
