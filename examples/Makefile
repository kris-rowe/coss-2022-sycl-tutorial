ifeq ($(CXX),g++)
	CXX:= clang++
endif
CXXFLAGS ?= -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(@)

default: all

0_device_info:
1_memory_management:
2_queues:
3_kernels:
4_events:
5_ranges:
6_reductions:
7_local_memory:
9_error_handling:

all: 0_device_info 1_memory_management 2_queues 3_kernels 4_events 5_ranges 6_reductions 7_local_memory 9_error_handling

clean:
	-rm -f 0_device_info
	-rm -f 1_memory_management
	-rm -f 2_queues
	-rm -f 3_kernels
	-rm -f 4_events
	-rm -f 5_ranges
	-rm -f 6_reductions
	-rm -f 7_local_memory
	-rm -f 9_error_handling

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"

.PHONY: all clean print-info
