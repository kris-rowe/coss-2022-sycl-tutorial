ifeq ($(CXX),g++)
	CXX:= clang++
endif
CXXFLAGS ?= -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $(@)


examples = \
00_device_info \
01_memory_management \
02_queues \
03_kernels \
04_events \
05_device_functions \
06_ranges	\
07_local_memory	\
08_reductions \
09_group_collectives \
10_error_handling 

all: $(examples)
default: all

clean:
	-rm -f $(examples)

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"

.PHONY: default all clean print-info
