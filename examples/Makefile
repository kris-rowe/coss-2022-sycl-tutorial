CXX ?= clang++
CXXFLAGS ?= -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(@)

default: all

0_device_info:
1_memory_management:
2_queues:
3_kernels:
4_events:
6_reductions:

all: 0_device_info 1_memory_management 2_queues 3_kernels 4_events 6_reductions

clean:
	-rm -f 0_device_info
	-rm -f 1_memory_management
	-rm -f 2_queues
	-rm -f 3_kernels
	-rm -f 4_events
	-rm -f 6_reductions

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"

.PHONY: all clean print-info
