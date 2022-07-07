#include <getopt.h>

#include <CL/sycl.hpp>
#include <exception>
#include <iostream>
#include <vector>

// Given a group of vectors of the same length, compute
// for (b=0; b < batch_size; ++b) {
//   Y += alpha * X + Y
// }
// where X is a vector located at offset stride_x * b in x
// and Y is a vector located at offset stride_Y * b in y
//
// N is the size of contiguous memory used to store the vectors
template <typename T>
sycl::event axpy_batch(sycl::queue& queue, int64_t N, T alpha, const T* x,
                       int64_t stride_x, T* y, int64_t stride_y,
                       int64_t batch_size,
                       const std::vector<sycl::event>& dependencies = {}) {
  if (N < batch_size * stride_x) {
    throw std::logic_error("N is smaller than batch_size * stride_x");
  }
  if (N < batch_size * stride_y) {
    throw std::logic_error("N is smaller than batch_size * stride_y");
  }

  sycl::event kernel_event = queue.parallel_for(
      {1, 1},  // Correct this. Remember, the last dimension is the "fastest".
      dependencies, [=](sycl::id<2> index) {
        // Find the batch index, and the index within the current vector
        // int batch_i = ???;
        // int i = ???;

        // Create pointers to the start of each vector for this axpy.
        // const T* batch_x = x + ???;
        // T* batch_y = y + ???;

        // Calculate axpy for the current vectors.
        // batch_y[i] += alpha * batch_x[i];
      });

  return kernel_event;
}

static struct option long_options[] {
  {"vector-size", required_argument, 0, 'N'}, {
    "batch-size", required_argument, 0, 'B'
  }
};

int main(int argc, char* argv[]) {
  // Default parameters
  size_t vector_size = 2000;
  size_t batch_size = 7;

  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "N:B:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'N':
        vector_size = std::stoul(optarg);
        break;
      case 'B':
        batch_size = std::stoul(optarg);
        break;
      default:
        std::cerr << "Usage: batch_axpy [-N vector-size] [-B batch-size]\n";
        exit(EXIT_FAILURE);
    }
  }

  std::cout << "Vector Size: " << vector_size << "\n";
  std::cout << "Batch Size: " << batch_size << "\n";

  const size_t total_size = vector_size * batch_size;
  std::vector<float> x_host(total_size);
  std::vector<float> y_host(total_size);
  const float alpha = 1.0;

  for (size_t b{}; b < batch_size; ++b) {
    for (size_t i{}; i < vector_size; ++i) {
      x_host[i + vector_size * b] = float(b);
      y_host[i + vector_size * b] = float(b);
    }
  }

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  float* x = sycl::malloc_device<float>(total_size, sycl_device, sycl_context);
  float* y = sycl::malloc_device<float>(total_size, sycl_device, sycl_context);

  sycl::event copy_x = sycl_queue.copy(x_host.data(), x, x_host.size());
  sycl::event copy_y = sycl_queue.copy(y_host.data(), y, y_host.size());

  sycl::event axpy_batch_kernel =
      axpy_batch(sycl_queue, total_size, alpha, x, vector_size, y, vector_size,
                 batch_size, {copy_x, copy_y});

  sycl_queue.copy(y, y_host.data(), y_host.size(), {axpy_batch_kernel}).wait();

  // Verify the results.
  size_t i{};
  for (const auto& y_i : y_host) {
    float expected = 2.0 * float(i / vector_size);
    if (expected != y_i) {
      std::cout << "Verification failed!\n";
      std::cout << "expected: " << expected << ", actual: " << y_i << std::endl;
      return EXIT_FAILURE;
    }
    ++i;
  }

  std::cout << "Success!\n";

  // Free device memory
  sycl::free(x, sycl_context);
  sycl::free(y, sycl_context);
  return EXIT_SUCCESS;
}
