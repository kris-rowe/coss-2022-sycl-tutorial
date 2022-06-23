#include <iostream>

#include "CL/sycl.hpp"

constexpr size_t megabyte{1024 * 1024};

void printAllocType(void* ptr, const sycl::context& sycl_context) {
  sycl::usm::alloc usm_alloc = sycl::get_pointer_type(ptr, sycl_context);
  switch (usm_alloc) {
    case sycl::usm::alloc::device:
      std::cout << "device allocation\n";
      break;
    case sycl::usm::alloc::host:
      std::cout << "host allocation\n";
      break;
    case sycl::usm::alloc::shared:
      std::cout << "shared allocation\n";
      break;
    default:
      std::cout << "unknown allocation?!\n";
  }
}

int main() {
  sycl::device sycl_device{sycl::default_selector()};
  sycl::context sycl_context{sycl_device};

  // Allocate 1 MB of device memory accessible, not accessible by the hose
  const size_t double_count = megabyte / sizeof(double);
  double* x =
      sycl::malloc_device<double>(double_count, sycl_device, sycl_context);

  // Allocate 1 MB of (page-locked) host memory accessible by the device
  constexpr size_t int_count = megabyte / sizeof(int);
  int* i = sycl::malloc_host<int>(int_count, sycl_context);

  // Allocate 1 MB of shared memory accessible by both the host and the device
  constexpr size_t float_count = megabyte / sizeof(float);
  float* y = sycl::malloc_shared<float>(float_count, sycl_device, sycl_context);

  // Check that the pointer belongs to the correct device.
  // Not needed here, but useful for libraries or complex applications.
  assert(sycl::get_pointer_device(x, sycl_context) == sycl_device);
  assert(sycl::get_pointer_device(i, sycl_context) == sycl_device);
  assert(sycl::get_pointer_device(y, sycl_context) == sycl_device);

  printAllocType(x, sycl_context);
  printAllocType(i, sycl_context);
  printAllocType(y, sycl_context);

  // Memory must be freed with the same context used for its allocation
  sycl::free(x, sycl_context);
  sycl::free(i, sycl_context);
  sycl::free(y, sycl_context);

  return 0;
}
