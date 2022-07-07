#ifndef _AXPY_HPP_
#define _AXPY_HPP_

#include <getopt.h>

#include <iostream>

namespace {

struct arguments_t {
  size_t N = 2000;
  size_t batch_size = 10;
};

arguments_t readArguments(int argc, char* argv[]) {
  static struct option long_options[] = {
      {"vector-size", required_argument, 0, 'N'},
      {"batch-size", required_argument, 0, 'B'}};

  arguments_t arguments;
  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "N:B:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'N':
        arguments.N = std::stoul(optarg);
        break;
      case 'B':
        arguments.batch_size = std::stoul(optarg);
        break;
      default:
        std::cerr << "Usage: batch_axpy [-N vector-size] [-B batch-size]\n";
        exit(EXIT_FAILURE);
    }
  }
  return arguments;
}

void printArguments(const arguments_t& arguments) {
  std::cout << "N: " << arguments.N << "\n";
  std::cout << "Batch Size: " << arguments.batch_size << "\n";
  std::cout << "\n";
}

}  // namespace

#endif
