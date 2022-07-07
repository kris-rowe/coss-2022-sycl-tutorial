#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "gemv.hpp"
#include "stats.hpp"

namespace {

// Naive implementation of GEMV function for verification purposes.
// Computes y = alpha * A(x) + beta * y
template <typename T>
void gemv(int64_t m, int64_t n, T alpha, const T* a, const T* x, T beta, T* y) {
  for (int64_t i = 0; i < m; ++i) {
    y[i] *= beta;
  }

  for (int64_t j = 0; j < n; ++j) {
    T x_j = x[j];
    for (int64_t i = 0; i < m; ++i) {
      // The sum of the columns of A weighted by alpha * x[j];
      y[i] += alpha * a[i + m * j] * x_j;
    }
  }
}

template <typename T>
sycl::event gemv(sycl::queue& sycl_queue, int64_t m, int64_t n, T alpha,
                 const T* a, const T* x, T beta, T* y,
                 const std::vector<sycl::event>& dependencies = {}) {
  sycl::range<1> kernel_range(m);
  sycl::event gemv_event =
      sycl_queue.parallel_for(kernel_range, dependencies, [=](sycl::id<1> i) {
        T y_i = beta * y[i];
        for (int64_t j = 0; j < n; ++j) {
          y_i += alpha * a[i + m * j] * x[j];
        }
        y[i] = y_i;
      });
  return gemv_event;
}

}  // namespace

int main(int argc, char* argv[]) {
  auto arguments = readArguments(argc, argv);
  printArguments(arguments);

  const size_t M = arguments.M;
  const size_t N = arguments.N;
  const size_t number_of_trials = arguments.trials;

  std::vector<float> x_host(N);
  std::vector<float> y_host(N);
  std::vector<float> A_host(M * N);

  std::random_device seed{};
  std::mt19937_64 generator{seed()};
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);

  const float alpha = distribution(generator);
  const float beta = distribution(generator);

  for (auto& x_i : x_host) x_i = distribution(generator);
  for (auto& y_i : y_host) y_i = distribution(generator);
  for (auto& A_ij : A_host) A_ij = distribution(generator);

  std::vector<float> y_valid = y_host;
  gemv(M, N, alpha, A_host.data(), x_host.data(), beta, y_valid.data());

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  float* x = sycl::malloc_device<float>(N, sycl_device, sycl_context);
  float* y = sycl::malloc_device<float>(M, sycl_device, sycl_context);
  float* A = sycl::malloc_device<float>(M * N, sycl_device, sycl_context);

  sycl::event copy_x = sycl_queue.copy(x_host.data(), x, N);
  sycl::event copy_y = sycl_queue.copy(y_host.data(), y, M);
  sycl::event copy_A = sycl_queue.copy(A_host.data(), A, M * N);

  sycl::event gemv_kernel =
      gemv(sycl_queue, M, N, alpha, A, x, beta, y, {copy_x, copy_y, copy_A});

  sycl_queue.copy(y, y_host.data(), M, {gemv_kernel}).wait();

  // Verify correctness
  for (int64_t i = 0; i < M; ++i) {
    if (std::abs(y_host[i] - y_valid[i]) > 1.0e-4f) {
      std::cout << "Verification failed!\n";
      std::cout << "expected: " << y_valid[i] << "\n";
      std::cout << "actual: " << y_host[i] << "\n";
      return EXIT_FAILURE;
    }
  }

  // Now run and time the kernel
  std::vector<double> times(number_of_trials);
  for (auto& runtime : times) {
    auto start_time = std::chrono::high_resolution_clock::now();
    gemv(sycl_queue, M, N, alpha, A, x, beta, y).wait();
    auto finish_time = std::chrono::high_resolution_clock::now();
    runtime =
        std::chrono::duration<double, std::milli>(finish_time - start_time)
            .count();
  }

  auto kernel_stats = stats::computeStats(times, "ms");
  std::cout << "Kernel Times\n";
  stats::printStats(kernel_stats);

  sycl::free(x, sycl_context);
  sycl::free(y, sycl_context);
  sycl::free(A, sycl_context);
  return EXIT_SUCCESS;
}
