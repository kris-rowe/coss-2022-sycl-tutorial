# 2022 Compute Ontario Summer School SYCL Tutorial

![build](https://github.com/kris-rowe/coss-2022-sycl-tutorial/workflows/build/badge.svg)

This repository contains notes and source code for the SYCL Tutorial presented virtually on July 8th, 2022 as part of the [Programming GPUs workshop](https://training.computeontario.ca/courses/enrol/index.php?id=11) during the [2022 Compute Ontario Summer School](https://training.computeontario.ca/index.php).

## Getting Started

### Requirements

- GNU Make
- SYCL 2020 compiler and libraries

> If a SYCL 2020 compiler is not installed on your current system, one can be built from [Intel's LLVM fork on GitHub](https://github.com/intel/llvm). Instructions for building and setting up the Intel LLVM compiler can be found [here](https://intel.github.io/llvm-docs/).

### Build

The [examples](examples/) and [exercises](examples/) directories contain makefiles to build their corresponding codes. By default, it is assumed that the LLVM clang compiler will be used with the CUDA plugin. 

Each example is contained in a single .cpp file, for which the makefile will generate an executable with the same name. Examples can be built individually, or all at once by calling 
```shell
$ make -j all
```

### Run

Example programs do not take commandline arguments and can be run by calling
```shell
$ ./example-name
```
For any additional instructions on running the exercise codes, see their corresponding README.

## Community

### Support

Need help? Ask a question in the [Q&A discussions category](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/q-a).

### Feedback

To provide feedback, participate in the [polls](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/polls) or start a conversation in the [ideas](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/ideas) discussion categories.

### Contributing

#### Bugs & Corrections

Found a bug, spelling mistake, or other error? Open an [issue](https://github.com/kris-rowe/coss-2022-sycl-tutorial/issues) and be sure to tag it with the corresponding category.

#### Sharing Your Work

Have an interesting solution to one of the exercises or other code related to the tutorial that you would like to share? Create a post in the [Show and tell](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/show-and-tell) discussions category.

#### Develoment

If you are interested in helping to further develop this tutorial please reach out to [Kris Rowe](mailto:kris.rowe@anl.gov).

### Code of Conduct

All discussion and other forms of participation related to this project should be consistent with [Argonne's Core Values](https://www.anl.gov/our-core-values) of respect, integrity, and teamwork.

## Acknowledgements

This work was supported by [Argonne Leadership Computing Facility](https://www.alcf.anl.gov), which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

## License

This project is available under a [MIT License](LICENSE.md)


