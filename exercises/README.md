# Exercises

Complete the exercises you are most comfortable with first. Feeling up for a challenge? Try tackling some of the more difficult tasks. Need help or want to know if you are on the right track? Ask a question in the [Q&A discussions category](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/q-a).

The [SYCL Reference Guide](https://www.khronos.org/files/sycl/sycl-2020-reference-guide.pdf) (cheat sheet) provides a concise summary of commonly used SYCL functions and is a helpful resource when first learning SYCL programming.

## 1. More Device Info

Extend the `device_info` example to provide more information about the available hardware. See the SYCL 2020 specification for a complete list of [device information descriptors](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_device_information_descriptors).

Some types of hardware support extensions that are not available otherwise. Extensions include support for certain floating-point types, atomic operations, or memory allocation types. These extensions can be queried through the function
```cpp
class device {
public:
  bool has(aspect asp) const
}
```
A list of [device aspects](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-aspects) (extensions) can be found in the SYCL specification.

Extend the `device_info` example to print whether a device supports `double` and `half` precision types.

## 2. Device Selection

A [sycl::device_selector](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:device-selector) is a function object which takes a `const sycl::device&` and returns an `int` score. When passed as an argument to the `sycl::device` constructor, the resulting device is the one with the highest score. If negative value is returned&mdash;for example, when no suitable device is found&mdash;a SYCL runtime exception is thrown.

Suppose that a hypothetical SYCL application targetting GPUs requires the use of `double` precision floating point numbers and 64-bit atomic operations. Create a `device_selector` which looks for such a device, and causes a runtime exception othersie. This can be accomplished by extending the `sycl::device_selector` class and implementing the member function `int operator()(const sycl::device& sycl_device) const`, as outlined in `02_device_selection.cpp`. 

> Note: You should handle the runtime exception if no suitable device can be found. If needed, see the `error_handling` example for a refresher exception handling.

## 3. Batched AXPY

Compute the BLAS axpy function for a group of vectors of the same length, stored consecutively in memory.

All of the necessary setup, cleanup, and verification is already implemented in `03_batch_axpy.cpp`. Following the source code comments, implement the body of the `axpy_batch` function. You will need to correct the global range for the kernel.

The vector and batch sizes can be passed as program arguments:
```shell
$ ./03_batch_axpy --vector-size N --batch-size B
```
Test the correctness of your implementation for different batch and vector sizes.
