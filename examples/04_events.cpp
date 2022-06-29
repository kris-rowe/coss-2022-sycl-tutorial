#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

int main() {
  const std::size_t vector_length = 2000;
  std::vector<double> a_host(vector_length);
  std::vector<double> b_host(vector_length, 1.0);
  std::vector<double> c_host(vector_length, 1.0);
  const double s = 1.0;

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate vectors on the device
  double* a =
      sycl::malloc_device<double>(vector_length, sycl_device, sycl_context);
  double* b =
      sycl::malloc_device<double>(vector_length, sycl_device, sycl_context);
  double* c =
      sycl::malloc_device<double>(vector_length, sycl_device, sycl_context);

  // Copy from the host to the device.
  sycl::event copy_b = sycl_queue.copy(b_host.data(), b, b_host.size());
  sycl::event copy_c = sycl_queue.copy(c_host.data(), c, c_host.size());

  // Submit work to the queue using a kernel defined via lambdas.
  sycl::event triad_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    // Use events returned from previous copies to express data dependencies.
    cgh.depends_on({copy_b, copy_c});
    cgh.parallel_for({vector_length}, [=](sycl::item<1> work_item) {
      int i = work_item.get_linear_id();
      a[i] = b[i] + s * c[i];
    });
  });

  // Copy from the device to the host. Use the event returned from submitting
  // the kernel to ensure this work is completed first.
  sycl::event copy_a =
      sycl_queue.copy(a, a_host.data(), a_host.size(), {triad_kernel});

  // Wait for copy to finish before verifying the results.
  copy_a.wait();

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
  return 0;
}