#include <getopt.h>

#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

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
  std::vector<double> runtimes(number_of_trials);

  const T alpha = 1.0;
  T* x = sycl::malloc_device<T>(N, sycl_queue);
  T* y = sycl::malloc_device<T>(N, sycl_queue);
  T* normy = sycl::malloc_device<T>(1, sycl_queue);

  sycl_queue.fill(x, T(1.0), N);
  sycl_queue.fill(y, T(1.0), N);
  sycl_queue.fill(normy, T(0.0), 1);
  sycl_queue.wait();

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
  
  sycl_queue.wait();
  sycl::free(x, sycl_queue);
  sycl::free(y, sycl_queue);
  sycl::free(normy, sycl_queue);

  return runtimes;
}

struct stats_t {
  double mean;
  double min;
  double max;
};

stats_t computeStats(std::vector<double>& dataset) {
  stats_t s;
  std::sort(dataset.begin(), dataset.end());
  double total = std::accumulate(dataset.begin(), dataset.end(), 0.0);
  s.mean = total / double(dataset.size());
  s.min = dataset.front();
  s.max = dataset.back();
  return s;
}

static struct option long_options[] {
  {"vector-size", required_argument, 0, 'N'}, {
    "number-of-trials", required_argument, 0, 'T'
  }
};

int main(int argc, char* argv[]) {
  const size_t megabyte = 1024 * 1024;
  size_t vector_size = megabyte / sizeof(float);
  size_t number_of_trials = 5000;

  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "N:T:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'N':
        vector_size = std::stoul(optarg);
        break;
      case 'T':
        number_of_trials = std::stoul(optarg);
        break;
      default:
        std::cerr
            << "Usage: kernel_fusion [-N vector-size] [-T number-of-trials]\n";
        exit(EXIT_FAILURE);
    }
  }

  std::cout << "Vector Size: " << vector_size << "\n";
  std::cout << "Number of Trials: " << number_of_trials << "\n";
  std::cout << "\n";

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  auto unfused_times =
      runBenchmark<float, false>(sycl_queue, vector_size, number_of_trials);
  auto fused_times =
      runBenchmark<float, true>(sycl_queue, vector_size, number_of_trials);

  auto unfused_stats = computeStats(unfused_times);
  auto fused_stats = computeStats(fused_times);

  std::cout.setf(std::ios::right);
  std::cout.precision(6);
  std::cout << "Unfused Kernel Times\n";
  std::cout << "--- mean: " << std::scientific << unfused_stats.mean << "ms\n";
  std::cout << "--- min: " << std::scientific << unfused_stats.min << "ms\n";
  std::cout << "--- max: " << std::scientific << unfused_stats.max << "ms\n";
  std::cout << "\n";
  std::cout << "Fused Kernel Times\n";
  std::cout << "--- mean: " << std::scientific << fused_stats.mean << "ms\n";
  std::cout << "--- min: " << std::scientific << fused_stats.min << "ms\n";
  std::cout << "--- max: " << std::scientific << fused_stats.max << "ms\n";

  return EXIT_SUCCESS;
}
