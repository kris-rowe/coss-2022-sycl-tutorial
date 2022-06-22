CXX := clang++
CXXFLAGS := -O2 -fsycl
SYCL_ROOT := $(CMPROOT)/linux

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(SYCL_ROOT)/include $< -o $(@) -L$(SYCL_ROOT)/lib -lsycl 

default: all

0_device_info:
1_memory_management:

all: 0_device_info 1_memory_management

clean:
	-rm -f 0_device_info
	-rm -f 1_memory_management

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SYCL_ROOT: $(SYCL_ROOT)"

.PHONY: all clean print-info
