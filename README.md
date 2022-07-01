# 2022 Compute Ontario Summer School SYCL Tutorial

![build](https://github.com/kris-rowe/coss-2022-sycl-tutorial/workflows/build/badge.svg)

This repository contains notes and source code for the SYCL Tutorial presented virtually on July 8th, 2022 as part of the *Programming GPUs* workshop during the [2022 Compute Ontario Summer School](https://training.computeontario.ca/index.php).

## Getting Started

### Requirements

- GNU Make
- C++17 compiler
- SYCL 2020 implementation

All required software already installed on Compute Ontario systems and Intel Devcloud as described below.

> If a SYCL 2020 implementation is not installed on your current system, one can be built from [Intel's LLVM fork on GitHub](https://github.com/intel/llvm). Instructions for building and setting up the Intel LLVM compiler can be found [here](https://intel.github.io/llvm-docs/).

#### Compute Ontario Systems

On Compute Ontario systems, a SYCL 2020 implementation is available through the [intel/2022.1.0 module](https://docs.alliancecan.ca/wiki/Available_software), which can be loaded using
```shell
$ module load intel/2022.1.0
```
Currently only the OpenCL CPU platform is available through this module, however it is possible to build and run all of the example codes.

#### Intel DevCloud

[Intel DevCloud](https://www.intel.com/content/www/us/en/developer/tools/devcloud/overview.html) provides free access to various Intel CPUs, GPUs, FPGAs, and other accelerators. New users can sign-up for access [here](https://www.intel.com/content/www/us/en/forms/idz/devcloud-registration.html?tgt=https://www.intel.com/content/www/us/en/secure/forms/devcloud-enrollment/account-provisioning.html). Once signed-up, follow the [instructions for connecting via ssh](https://devcloud.intel.com/oneapi/documentation/connect-with-ssh-windows-cygwin/#configure-ssh-connection).

To clone the tutorial from GitHub and build example codes, it is first necessary to launch a job on one of the compute nodes. For example, an interactive session on a GPU compute node can be started with the command
```shell
$ qsub -I -l nodes=1:gpu:ppn=2
```
Various software packages are provided through [environment modules](https://devcloud.intel.com/oneapi/documentation/modules/). The latest Intel oneAPI toolkit can be loaded by calling
```shell
$ module load /glob/module-files/intel-oneapi/latest
```

### Build

The [examples](examples/) and [exercises](exercises/) directories contain makefiles to build their corresponding codes. By default, it is assumed that the Intel LLVM C++ compiler&mdash;named `icpx`&mdash;will be used. 

Each example is contained in a single .cpp file, for which the makefile will generate an executable with the same name. Examples can be built individually, or all at once by calling 
```shell
$ make -j all
```

### Run

Example programs do not take commandline arguments and can be run by calling
```shell
$ ./example-name
```
For any additional instructions on running the exercise codes, see the corresponding README.

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

#### Development

If you are interested in helping to further develop this tutorial please reach out to [Kris Rowe](mailto:kris.rowe@anl.gov).

### Code of Conduct

All discussion and other forms of participation related to this project should be consistent with [Argonne's Core Values](https://www.anl.gov/our-core-values) of respect, integrity, and teamwork.

## Acknowledgements

This work was supported by [Argonne Leadership Computing Facility](https://www.alcf.anl.gov), which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

## License

This project is available under a [MIT License](LICENSE.md)


