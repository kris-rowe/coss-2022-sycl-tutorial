#include <CL/sycl.hpp>
#include <iostream>

constexpr int gigabyte{1024 * 1024 * 1024};

void printDeviceType(sycl::device& sycl_device) {
  sycl::info::device_type device_type =
      sycl_device.get_info<sycl::info::device::device_type>();
  std::cout << "Type: ";
  switch (device_type) {
    case sycl::info::device_type::cpu:
      std::cout << "CPU\n";
      break;
    case sycl::info::device_type::gpu:
      std::cout << "GPU\n";
      break;
    case sycl::info::device_type::accelerator:
      std::cout << "accelerator\n";
      break;
    case sycl::info::device_type::host:
      std::cout << "host\n";
      break;
    default:
      std::cout << "???\n";
  }
}

int main() {
  auto platforms = sycl::platform::get_platforms();
  int platform_id = 0;

  for (auto& p : platforms) {
    std::string platform_name = p.get_info<sycl::info::platform::name>();
    std::cout << "Platform " << platform_id << ": " << platform_name << "\n";

    auto devices = p.get_devices();
    int device_id = 0;
    for (auto& d : devices) {
      std::string device_name = d.get_info<sycl::info::device::name>();
      std::cout << "Device " << device_id << ": " << device_name << "\n";

      printDeviceType(d);

      int memory = d.get_info<sycl::info::device::global_mem_size>();
      std::cout << "Memory: " << (memory / gigabyte) << " GB\n";

      uint64_t max_wg_size =
          d.get_info<sycl::info::device::max_work_group_size>();
      std::cout << "Max Work Group Size: " << max_wg_size << "\n";

      //----------
      // Find and print out other device info here.

      //----------
      ++device_id;
    }
    ++platform_id;
    std::cout << "\n";
  }

  return EXIT_SUCCESS;
}
