ifeq ($(CXX),g++)
	CXX:= clang++
endif
CXXFLAGS ?= -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(@)

default: all

examples = \
0_device_info \
1_memory_management \
2_queues	\
3_kernels	\
4_events \
5_ranges	\
6_reductions \
7_local_memory	\
9_error_handling 

$(examples): ;

all: $(examples)

clean:
	-rm -f $(examples)

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"

.PHONY: all clean print-info
