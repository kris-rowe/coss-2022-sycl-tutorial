#include <vector>

#include "CL/sycl.hpp"

// Aliasing oneAPI DPC++ specific extensions
namespace dpcpp = sycl::ext::oneapi;

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

  // Do block multiplication using 16x16 blocks.
  constexpr int block_size{16};
  constexpr int tile_size{8};

  // Define a 2D range for the matrix multiplication kernel
  sycl::range<2> local_range(block_size, block_size);
  sycl::range<2> global_range(N, M);
  sycl::nd_range<2> kernel_range(global_range, local_range);

  // Submit work to the queue using a kernel defined via lambdas.
  sycl::event gemm_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({copy_a, copy_b});
    cgh.parallel_for(kernel_range, [=](sycl::nd_item<2> work_item) {
      // The last dimension of an ND-range is the "fastest"
      int i = work_item.get_local_id(1);
      int j = work_item.get_local_id(0);

      int i_global = work_item.get_global_id(1);
      int j_global = work_item.get_global_id(0);

      auto work_group = work_item.get_group();

      // Allocate SLM to use as an explicit cache
      using tile_t = float[tile_size][block_size];
      tile_t& A_tile =
          *dpcpp::group_local_memory_for_overwrite<tile_t>(work_group);
      tile_t& B_tile =
          *dpcpp::group_local_memory_for_overwrite<tile_t>(work_group);

      // Compute C = A * B
      // Each work-group will compute a 16x16 block of C
      // Tile the k-loop by a factor of 8
      float C_ij{};
      for (int k_tile{}; k_tile < K; k_tile += tile_size) {
        // Here j plays the role of k
        if (j < tile_size) {
          // Load one tile of A from global to shared local memory
          A_tile[j][i] = A[i_global + M * (k_tile + j)];
        }

        // Here i plays the role k
        if (i < tile_size) {
          // Load one tile of B from global to shared local memory
          B_tile[i][j] = B[(i + k_tile) + K * j_global];
        }

        // Synchronize the work-group since we wrote to SLM
        sycl::group_barrier(work_group);

        for (int k = 0; k < tile_size; ++k) {
          // Do matrix-matrix multiplication using the tiles
          // Each work-item computes one entry of the output
          C_ij += A_tile[k][i] * B_tile[k][j];
        }

        // Synchronize the work-group since we read from SLM
        sycl::group_barrier(work_group);
      }

      // Each work-item writes its result back to global memory
      C[i_global + M * j_global] = C_ij;
    });
  });

  // Copy the result from the device to the host; synchronize.
  sycl_queue.copy(C, C_host.data(), C_host.size(), {gemm_kernel}).wait();

  // Verify the results
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
