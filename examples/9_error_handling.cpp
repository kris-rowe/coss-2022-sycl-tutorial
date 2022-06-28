#include <iostream>

#include "CL/sycl.hpp"

int main() {
  try {
    // Create a host device
    sycl::device sycl_device{sycl::default_selector()};
    sycl::context sycl_context{sycl_device};
    sycl::queue sycl_queue{sycl_context, sycl_device};

    int i{};
    // Mock device_malloc failing and returning nullptr
    int* j = nullptr;
    sycl_queue.copy<int>(&i, j, 1);

  } catch (const sycl::exception& e) {
    std::cerr << "Synchronous exception\n";
    std::cerr << e.what() << std::endl;
  }
  std::cout << "Exception handled.\n";
  return 0;
}
