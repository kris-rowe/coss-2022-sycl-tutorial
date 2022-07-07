#include <getopt.h>

#include <CL/sycl.hpp>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

struct arguments_t {
  size_t M = 1024;
  size_t N = 1024;
  size_t trials = 100;
};

arguments_t readArguments(int argc, char* argv[]) {
  static struct option long_options[] = {
    {"rows", required_argument, 0, 'M'},
    {"columns", required_argument, 0, 'N'},
    {"trials", required_argument, 0, 'T'}};

  arguments_t arguments;
  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "M:N:T:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'M':
        arguments.M = std::stoul(optarg);
        break;
      case 'N':
        arguments.N = std::stoul(optarg);
        break;
      case 'T':
        arguments.trials = std::stoul(optarg);
      default:
        std::cerr << "Usage: gemv_part1 [-M or --rows nrows] [-N or --columns "
                     "ncolumns] [-T or --trials ntrials] \n";
        exit(EXIT_FAILURE);
    }
  }
  std::cout << std::right;
  std::cout << std::setw(10) << "GEMV\n";
  std::cout << std::setw(10) << "M: " << arguments.M << "\n";
  std::cout << std::setw(10) << "N: " << arguments.N << "\n";
  std::cout << std::setw(10) << "Trials: " << arguments.trials << "\n";

  return arguments;
}

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

int main(int argc, char* argv[]) {
  auto arguments = readArguments(argc, argv);
  const size_t M = arguments.M;
  const size_t N = arguments.N;
  const size_t numer_of_trials = arguments.trials;

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

  sycl_queue.copy(x_host.data(), x, N);
  sycl_queue.copy(y_host.data(), y, M);
  sycl_queue.copy(A_host.data(), A, M * N);
  sycl_queue.wait();

  gemv(sycl_queue, M, N, alpha, A, x, beta, y);
  sycl_queue.wait();

  sycl_queue.copy(y, y_host.data(), M);
  sycl_queue.wait();

  for (int64_t i = 0; i < M; ++i) {
    if (std::abs(y_host[i] - y_valid[i]) > 1.0e-4f) {
      std::cout << "Verification failed!\n";
      std::cout << "expected: " << y_valid[i] << "\n";
      std::cout << "actual: " << y_host[i] << "\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  sycl::free(x, sycl_context);
  sycl::free(y, sycl_context);
  sycl::free(A, sycl_context);
  return EXIT_SUCCESS;
}
