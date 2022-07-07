#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "stats.hpp"
#include "fusion.hpp"

namespace {

template <typename T>
sycl::event axpyDot(sycl::queue& sycl_queue, int64_t N, T alpha, const T* x,
                    T* y, T* normy) {
  sycl::range<1> kernel_range(N);

  // First compute axpy
  sycl::event axpy_event = sycl_queue.parallel_for(
      kernel_range, [=](sycl::id<1> i) { y[i] += alpha * x[i]; });

  // Next compute dot y
  sycl::event normy_event = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on(axpy_event);

    auto reduce_normy = sycl::reduction(normy, sycl::plus<>());
    cgh.parallel_for(kernel_range, reduce_normy,
                     [=](sycl::id<1> it, auto& normy_) {
                       size_t i = it;
                       if (i < N) normy_ += y[i] * y[i];
                     });
  });

  return normy_event;
}

template <typename T>
sycl::event axpyDotFused(sycl::queue& sycl_queue, int64_t N, T alpha,
                         const T* x, T* y, T* normy) {
  sycl::range<1> kernel_range(N);

  // Compute axpy and then dot in the same kernel
  sycl::event kernel_event = sycl_queue.submit([&](sycl::handler& cgh) {
    auto reduce_normy = sycl::reduction(normy, sycl::plus<>());
    cgh.parallel_for(kernel_range, reduce_normy,
                     [=](sycl::id<1> i, auto& normy_) {
                       T y_i = alpha * x[i] + y[i];
                       y[i] = y_i;
                       normy_ += y_i * y_i;
                     });
  });
  return kernel_event;
}

template <typename T, bool is_fused>
std::vector<double> runBenchmark(sycl::queue& sycl_queue, int64_t N,
                                 size_t number_of_trials) {
  const T alpha = 1.0;
  T* x = sycl::malloc_device<T>(N, sycl_queue);
  T* y = sycl::malloc_device<T>(N, sycl_queue);
  T* normy = sycl::malloc_device<T>(1, sycl_queue);

  sycl_queue.fill(x, T(1.0), N);
  sycl_queue.fill(y, T(1.0), N);
  sycl_queue.fill(normy, T(0.0), 1);
  sycl_queue.wait();

  std::vector<double> runtimes(number_of_trials);
  for (auto& runtime : runtimes) {
    auto start_time = std::chrono::high_resolution_clock::now();
    if (!is_fused) {
      axpyDot(sycl_queue, N, alpha, x, y, normy).wait();
    } else {
      axpyDotFused(sycl_queue, N, alpha, x, y, normy).wait();
    }

    auto finish_time = std::chrono::high_resolution_clock::now();
    runtime =
        std::chrono::duration<double, std::milli>(finish_time - start_time)
            .count();
  }

  sycl::free(x, sycl_queue);
  sycl::free(y, sycl_queue);
  sycl::free(normy, sycl_queue);

  return runtimes;
}

}

int main(int argc, char* argv[]) {
  auto arguments = readArguments(argc,argv);
  printArguments(arguments);
  
  const size_t N = arguments.N;
  const size_t number_of_trials = arguments.trials;

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  auto unfused_times =
      runBenchmark<float, false>(sycl_queue, N, number_of_trials);
  auto fused_times =
      runBenchmark<float, true>(sycl_queue, N, number_of_trials);

  auto unfused_stats = stats::computeStats(unfused_times, "ms");
  auto fused_stats = stats::computeStats(fused_times, "ms");

  std::cout << "Unfused Kernel Times\n";
  stats::printStats(unfused_stats);
  std::cout << "Fused Kernel Times\n";
  stats::printStats(fused_stats);

  return EXIT_SUCCESS;
}
