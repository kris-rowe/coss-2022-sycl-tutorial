#include <vector>

#include "CL/sycl.hpp"

int main() {
  constexpr size_t M{256};
  constexpr size_t N{128};
  constexpr size_t K{64};

  // Linear arrays to store matrices
  std::vector<float> A_host(M * K, 1.0);
  std::vector<float> B_host(K * N, 1.0);
  std::vector<float> C_host(M * N);

  // Create a sycl::queue using the default device selector
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  // Allocate matrices on the device
  float* A =
      sycl::malloc_device<float>(A_host.size(), sycl_device, sycl_context);
  float* B =
      sycl::malloc_device<float>(B_host.size(), sycl_device, sycl_context);
  float* C =
      sycl::malloc_device<float>(C_host.size(), sycl_device, sycl_context);

  // Copy from the host to the device
  sycl::event copy_a = sycl_queue.copy(A_host.data(), A, A_host.size());
  sycl::event copy_b = sycl_queue.copy(B_host.data(), B, B_host.size());

  sycl::event gemm_kernel =
      sycl_queue.parallel_for({N, M}, {copy_a, copy_b}, [=](sycl::id<2> ij) {
        // In SYCL the last dimension is alays the "fastest"
        int i = ij[1];
        int j = ij[0];

        // Compute C = A * B
        float C_ij{};
        for (int k = 0; k < K; ++k) {
          C_ij += A[i + M * k] * B[k + K * j];
        }
        C[i + M * j] = C_ij;
      });

  // Copy the result from the device to the host; synchronize.
  sycl_queue.copy(C, C_host.data(), C_host.size(), {gemm_kernel}).wait();

  // Verify the results.
  for (const auto& C_ij : C_host) {
    if (static_cast<float>(K) != C_ij) {
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
