#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

int main() {
  const int vector_length = 2000;
  std::vector<float> x_host(vector_length, 1.0);
  std::vector<float> y_host(vector_length, 1.0);

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate vectors on the device
  float* x =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);
  float* y =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);
  float* xdoty = sycl::malloc_device<float>(1, sycl_device, sycl_context);

  // Copy from the host to the device.
  sycl::event copy_x = sycl_queue.copy(x_host.data(), x, x_host.size());
  sycl::event copy_y = sycl_queue.copy(y_host.data(), y, y_host.size());

  // For portability, find the maximum work-group size of the device.
  const int work_group_size = 256;
  // The global range must be divisible by the work-group size.
  // Compute the ceiling function to determine an appropriate range.
  const int work_group_count =
      (vector_length + work_group_size - 1) / work_group_size;

  sycl::range<1> global_range(work_group_size * work_group_count);
  sycl::range<1> local_range(work_group_size);
  sycl::nd_range<1> kernel_range(global_range, local_range);

  // Submit work to the queue using a kernel defined via lambdas.
  sycl::event dotp_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({copy_x, copy_y});
    auto reduce_xdoty = sycl::reduction(xdoty, sycl::plus<>());

    cgh.parallel_for(kernel_range, reduce_xdoty,
                     [=](auto work_item, auto& xdoty_) {
                       int i = work_item.get_global_linear_id();
                       if (i < vector_length) xdoty_ += x[i] * y[i];
                     });
  });

  // Copy result from device to host; synchronize;
  float xdoty_host;
  sycl_queue.copy(xdoty, &xdoty_host, 1, {dotp_kernel}).wait();

  // Verify the results.
  if (static_cast<float>(vector_length) != xdoty_host) {
    std::cout << "Verification failed!\n";
    return EXIT_FAILURE;
  }

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(x, sycl_context);
  sycl::free(y, sycl_context);
  sycl::free(xdoty, sycl_context);
  return EXIT_SUCCESS;
}
