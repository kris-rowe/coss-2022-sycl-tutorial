#include <vector>

#include "CL/sycl.hpp"

int main() {
  constexpr int M{256};
  constexpr int N{128};
  constexpr int K{64};

  // Linear arrays to store matrices
  std::vector<double> A_host(M * K, 1.0);
  std::vector<double> B_host(K * N, 1.0);
  std::vector<double> C_host(M * N);

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate matrices on the device
  double* A =
      sycl::malloc_device<double>(A_host.size(), sycl_device, sycl_context);
  double* B =
      sycl::malloc_device<double>(B_host.size(), sycl_device, sycl_context);
  double* C =
      sycl::malloc_device<double>(C_host.size(), sycl_device, sycl_context);

  // Copy from the host to the device
  sycl::event copy_a = sycl_queue.copy(A_host.data(), A, A_host.size());
  sycl::event copy_b = sycl_queue.copy(B_host.data(), B, B_host.size());

  // Define a 2D range for the matrix multiplication kernel
  // Do block multiplication using 16x16 blocks.
  sycl::range<2> local_range(16, 16);
  sycl::range<2> global_range(N, M);
  sycl::nd_range<2> kernel_range(global_range, local_range);

  // Submit work to the queue using a kernel defined via lambdas.
  sycl::event gemm_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({copy_a, copy_b});
    cgh.parallel_for(kernel_range, [=](sycl::nd_item<2> work_item) {
      // The last dimension of an ND-range is the "fastest"
      int i = work_item.get_global_id(1);
      int j = work_item.get_global_id(0);

      // Compute C = A * B
      // Each work-group will compute a 16x16 block of C
      double C_ij{};
      for (int k = 0; k < K; ++k) {
        C_ij += A[i + M * k] * B[k + K * j];
      }
      C[i + M * j] = C_ij;
    });
  });

  // Copy the result from the device to the host; synchronize.
  sycl_queue.copy(C, C_host.data(), C_host.size(), {gemm_kernel}).wait();

  // Verify the results
  // Verify the results.
  for (const auto& C_ij : C_host) {
    if (static_cast<double>(K) != C_ij) {
      std::cout << "Verification failed!\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  sycl::free(A, sycl_context);
  sycl::free(B, sycl_context);
  sycl::free(C, sycl_context);
  return EXIT_SUCCESS;
}
