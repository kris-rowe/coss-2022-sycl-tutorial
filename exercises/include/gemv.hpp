#ifndef _GEMV_H_
#define _GEMV_H_
#include <getopt.h>

#include <iostream>

namespace {

struct arguments_t {
  size_t M = 1024;
  size_t N = 1024;
  size_t trials = 5000;
};

arguments_t readArguments(int argc, char* argv[]) {
  static struct option long_options[] = {{"rows", required_argument, 0, 'M'},
                                         {"columns", required_argument, 0, 'N'},
                                         {"trials", required_argument, 0, 'T'}};

  arguments_t arguments;
  while (1) {
    int option_index{};
    int c = getopt_long(argc, argv, "M:N:T:", long_options, &option_index);
    if (0 > c) break;

    switch (c) {
      case 'M':
        arguments.M = std::stoul(optarg);
        break;
      case 'N':
        arguments.N = std::stoul(optarg);
        break;
      case 'T':
        arguments.trials = std::stoul(optarg);
      default:
        std::cerr << "Usage: gemv_part1 [-M or --rows nrows] [-N or --columns "
                     "ncolumns] [-T or --trials ntrials] \n";
        exit(EXIT_FAILURE);
    }
  }
  return arguments;
}

void printArguments(const arguments_t& arguments) {
  std::cout << "M: " << arguments.M << "\n";
  std::cout << "N: " << arguments.N << "\n";
  std::cout << "Trials: " << arguments.trials << "\n";
  std::cout << "\n";
}

}  // namespace
#endif
