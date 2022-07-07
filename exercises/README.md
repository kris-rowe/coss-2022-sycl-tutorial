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

Suppose that a hypothetical SYCL application targetting GPUs requires the use of `double` precision floating point numbers and 64-bit atomic operations. Create a `device_selector` which looks for such a device, and causes a runtime exception otherwise. This can be accomplished by extending the `sycl::device_selector` class and implementing the member function `int operator()(const sycl::device& sycl_device) const`, as outlined in `02_device_selection.cpp`. 

> Note: You should handle the runtime exception if no suitable device can be found. If needed, see the `error_handling` example for a refresher exception handling.

## 3. Batched AXPY

Compute the BLAS axpy function for a group of vectors of the same length, stored consecutively in memory.

All of the necessary setup, cleanup, and verification is already implemented in `03_batch_axpy.cpp`. Following the source code comments, implement the body of the `axpy_batch` function. You will need to correct the global range for the kernel.

The vector and batch sizes can be passed as program arguments:
```shell
$ ./03_batch_axpy --vector-size N --batch-size B
```
Test the correctness of your implementation for different batch and vector sizes.

## 4. Kernel Fusion

 Kernel Fusion combines the logic for two or more kernels into a single kernel&mdash;by directly merging source code or using advanced programming techniques&mdash;to avoid extra kernel launches and trips through the memory hierarchy.

Often, this strategy can improve performance when the results of a *compute bound* kernel (think BLAS 2/3) are used by a *memory bound* kernel (think BLAS 1) which immediately follows. A common example of where this occurs is in iterative solvers for linear-systems, such as the conjugate gradient method.

The program given in `04_kernel_fusion.cpp` computes the BLAS axpy function for two vectors, followed by the squared norm of the result. Fused and unfused kernels are run a fixed number of iterations to obtain runtime statitistcs.

The vector size and number of trials can be passed as program arguments:
```shell
$ ./04_kernel_fusion --vector-size N --number-of-trials T
```

Perform a series of experiments, running the `kernel_fusion` benchmark for a range of vector sizes&mdash;e.g., between 2^18 (1 MB) and 2^28 (1 GB). Plot the mean runtime against the vector size for both the fused and unfused kernels. For which vector sizes does kernel fusion provide the most benefit? Can you explain the observed behaviour in the limit of small vector sizes? large vector sizes?

## 5. GEMV

The program `05_gemv` benchmarks the performance of a kernel implementing the BLAS gemv function, which calculates dense matrix-vector products. Input data is initialized using pseudorandom values. First the correctness of the kernel is verified using a naive host-side gemv function. Then, the device kernel is run for a fixed number of iterations to obtain runtime statistics.

The matrix dimensions and number of trials can be passed as program arguments:
```shell
$ ./05_gemv --rows M --columns N --number-of-trials T
```

A provided kernel contains a basic implementation of gemv, but is not very performant. Performance can be improved via shared local memory and/or using group collectives. However, these features can only be used with `nd_range` kernels.

Transform the provided basic kernel into an `nd_range` kernel. You will need to choose an appropriate work-group size&mdash;for example, using heuristics or by querying device information. Recall that [all work-groups in a `parallel_for` must be the same size](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_work_group_data_parallel_kernels), implying that the global size of an `nd_range` must be a multiple of the work-group size in each dimension. Therefore, you will need to address the most common case where matrix/vector dimensions are not divisble by the work-group size. This can be accomplished by a second kernel launch to address the remainder "loop" (range), or conditional checks within a single kernel launch.

After verifying the correctnes of your `nd_range` kernel, consider how to improve kernel performance using a cache-blocking technique. Allocate shared local memory (see [example \#7](../examples/07_local_memory.cpp)) and explicitly cache any matrix or vector tiles. Introduce group barriers where appropriate to avoid any data-race conditions&mdash;e.g., after writing to or reading from SLM.

Run the `gemv` benchmark for different problem sizes using the provided basic kernel and your `nd_range` implementation. How does the performance of your new kernel compare with the original? For what problem sizes does data caching provide the greatest benefit? Experiment with different work-group sizes. Which work-group sizes lead to the best performance? (*Hint: on NVIDIA hardware think about multiples of 32*)

### Challenge: Group Collectives

Can you implement a similar tiled gemv `nd_range` kernel *without using shared local memory*? To accomplish this, you will need to use [group collectives](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:group-functions) to communicate data private to each work-item with other work-items in the same group or sub-group. Compare the performance of your new kernel with your `nd_range` kernel which used SLM.

> If you complete this challenge exercise and would like to show-off your work, create a post in the [Show and tell discussions category](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/show-and-tell).
