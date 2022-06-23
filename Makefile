CXX ?= clang++
CXXFLAGS ?= -O2 -fsycl

%.exe: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(@) 

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
