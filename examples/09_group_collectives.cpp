#include <random>
#include <vector>

#include "CL/sycl.hpp"

int main() {
  int number_of_nodes = 8;
  int number_of_elements = 1000;
  int nodes_per_element = number_of_nodes * number_of_nodes;
  int boundary_per_element = 4 * (number_of_nodes - 1);
  int total_nodes = number_of_elements * nodes_per_element;
  int total_boundary = number_of_elements * boundary_per_element;

  std::vector<int> u_host(total_nodes);
  std::vector<int> ub_host(total_boundary);
  std::vector<int> ub_valid(total_boundary);

  int b = 0;
  for (int e{}; e < number_of_elements; ++e) {
    for (int j{}; j < number_of_nodes; ++j) {
      for (int i{}; i < number_of_nodes; ++i) {
        int index = i + number_of_nodes * j + nodes_per_element * e;
        u_host[index] = index;
        bool on_boundary = (i == 0) || (i == (number_of_nodes - 1)) ||
                           (j == 0) || (j == (number_of_nodes - 1));
        if (on_boundary) {
          ub_valid[b] = index;
          ++b;
        }
      }
    }
  }

  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};
  sycl::queue sycl_queue{sycl_context, sycl_device};

  int* u = sycl::malloc_device<int>(total_nodes, sycl_device, sycl_context);
  int* ub = sycl::malloc_device<int>(total_boundary, sycl_device, sycl_context);

  sycl::event copy_u = sycl_queue.copy(u_host.data(), u, total_nodes);

  sycl::range<2> local_range(1, nodes_per_element);
  sycl::range<2> global_range(number_of_elements, nodes_per_element);
  sycl::nd_range<2> kernel_range(global_range, local_range);

  sycl::event boundary_kernel = sycl_queue.submit([&](sycl::handler& cgh) {
    cgh.depends_on({copy_u});
    cgh.parallel_for(kernel_range, [=](sycl::nd_item<2> work_item) {
      int e = work_item.get_global_id(0);
      int ij = work_item.get_local_id(1);

      int i = ij % number_of_nodes;
      int j = ij / number_of_nodes;
      int on_boundary = (i == 0) || (i == (number_of_nodes - 1)) || (j == 0) ||
                        (j == (number_of_nodes - 1));

      auto wg = work_item.get_group();

      // Find the offset to the next boundary point within the group
      int b = sycl::exclusive_scan_over_group(wg, on_boundary, sycl::plus<>());
      if (on_boundary) {
        ub[b + boundary_per_element * e] = u[ij + nodes_per_element * e];
      }
    });
  });

  // Copy the result from the device to the host; synchronize.
  sycl_queue.copy(ub, ub_host.data(), total_boundary, {boundary_kernel}).wait();

  // Verify the results
  for (int b{}; b < total_boundary; ++b) {
    if (ub_valid[b] != ub_host[b]) {
      std::cout << "Verification failed!\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "Success!\n";

  sycl::free(u, sycl_context);
  sycl::free(ub, sycl_context);
  return EXIT_SUCCESS;
}
