CXX := clang++
CXXFLAGS := -O2 -fsycl
SYCL_ROOT := $(CMPROOT)/linux

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) -I$(SYCL_ROOT)/include $< -o $(@) -L$(SYCL_ROOT)/lib -lsycl 

default: all

all: 0_device_info

clean:
	-rm -f 0_device_info

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SYCL_ROOT: $(SYCL_ROOT)"

.PHONY: all clean print-info
