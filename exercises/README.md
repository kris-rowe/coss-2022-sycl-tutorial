# Exercises

Each exercise is divided into one or more parts of increasing difficulty. The tasks described in the *Dip Your Toes* section are relatively straightforward to complete, revisiting ideas covered in the examples. Tasks in the *Wade-in* section build on previously covered concepts and, while requiring more thought, are still meant to be approachable. The *Dive Deep* section offers the opportunity for self-directed learning by providing references to more advanced concepts and a context in which they can be applied.

Complete the exercises you are most comfortable with first. Feeling up for a challenge? Try tackling some of the more difficult tasks. Need help or want to know if you are on the right track? Ask a question in the [Q&A discussions category](https://github.com/kris-rowe/coss-2022-sycl-tutorial/discussions/categories/q-a).

## 1. More Device Info

### Dip Your Toes

Extend the `device_info` example to provide more information about the available hardware. See the SYCL 2020 specification for a complete list of [device information descriptors](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_device_information_descriptors).

### Wade-in

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
