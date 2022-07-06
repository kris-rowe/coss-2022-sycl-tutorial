#include <iostream>

#include "CL/sycl.hpp"

struct GPUWithFP64AtomicsSelector : sycl::device_selector {
  int operator()(const sycl::device& sycl_device) const {
    // If no suitable device can be found, cause an exception
    return -1;
  }
};

int main() {
  sycl::device gpu_with_fp64_atomics{GPUWithFP64AtomicsSelector()};
  auto device_name = gpu_with_fp64_atomics.get_info<sycl::info::device::name>();
  std::cout << "Selected " << device_name << "\n";

  std::cout << "Success\n";
  return EXIT_SUCCESS;
}
