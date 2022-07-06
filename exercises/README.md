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

## 2. 

## 3. Peer-to-peer Copy

This exercise extends the `queues` example and demonstrates the copying of memory between two GPU devices. Follow the source-code comments in `03_p2p_copy.cpp`.