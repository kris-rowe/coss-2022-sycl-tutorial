#ifndef _FUSION_HPP_
#define _FUSION_HPP_

#include <getopt.h>

#include <iostream>

namespace {

struct arguments_t {
  size_t N = 262144;
  size_t trials = 5000;
};

arguments_t readArguments(int argc, char* argv[]) {
  static struct option long_options[] = {
      {"vector-size", required_argument, 0, 'N'},
      {"trials", required_argument, 0, 'T'}};

  arguments_t arguments;
  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "N:T:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'N':
        arguments.N = std::stoul(optarg);
        break;
      case 'T':
        arguments.trials = std::stoul(optarg);
        break;
      default:
        std::cerr << "Usage: kernel_fusion [-N vector-size] [-T trials]\n";
        exit(EXIT_FAILURE);
    }
  }
  return arguments;
}

void printArguments(const arguments_t& arguments) {
  std::cout << "N: " << arguments.N << "\n";
  std::cout << "Trials: " << arguments.trials << "\n";
  std::cout << "\n";
}

}  // namespace

#endif
