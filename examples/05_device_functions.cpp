#include <iostream>
#include <vector>

#include "CL/sycl.hpp"

template <typename T>
void axpy(int N, T alpha, const T* x, T* y) {
  for (int i{}; i < N; ++i) {
    y[i] += alpha * x[i];
  }
}

int main() {
  const size_t vector_length = 2000;
  std::vector<float> x_host(vector_length, 1.0);
  std::vector<float> y_host(vector_length, 0.0);
  const float alpha = 1.0;

  // Here we are calling the function from the host,
  // so it has been compiled as regular C++ function
  axpy(vector_length, alpha, x_host.data(), y_host.data());

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  float* x =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);
  float* y =
      sycl::malloc_device<float>(vector_length, sycl_device, sycl_context);

  sycl::event copy_x = sycl_queue.copy(x_host.data(), x, x_host.size());
  sycl::event copy_y = sycl_queue.copy(y_host.data(), y, y_host.size());

  // Each work-item will compute 4 entries
  constexpr int thread_vector_length{4};

  sycl::event axpy_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({copy_x, copy_y});

    cgh.parallel_for({vector_length / thread_vector_length},
                     [=](sycl::item<1> work_item) {
                       // These are private to each work-item
                       float x_thread[thread_vector_length];
                       float y_thread[thread_vector_length];

                       int i = work_item.get_linear_id();
                       int r = work_item.get_range(0);

                       for (int n{}; n < thread_vector_length; ++n) {
                         x_thread[n] = x[i + r * n];
                         y_thread[n] = y[i + r * n];
                       }

                       // Since we are calling the function inside a kernel,
                       // it was compiled for the device.
                       axpy(thread_vector_length, alpha, x_thread, y_thread);

                       for (int n{}; n < thread_vector_length; ++n) {
                         y[i + r * n] = y_thread[n];
                       }
                     });
  });

  sycl_queue.copy(y, y_host.data(), y_host.size(), {axpy_kernel}).wait();

  // Verify the results.
  for (const auto& y_i : y_host) {
    if (2.0 != y_i) {
      std::cout << "Verification failed!\n";
      std::cout << "expected: 2.0, actual: " << y_i << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(x, sycl_context);
  sycl::free(y, sycl_context);
  return EXIT_SUCCESS;
}
