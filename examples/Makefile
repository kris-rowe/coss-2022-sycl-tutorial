CXX := clang++
CXXFLAGS := -O2 -std=c++17
SYCLFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

programs = 00_device_info 01_memory_management 02_queues 03_kernels \
04_events 05_device_functions 06_ranges	07_local_memory	08_reductions \
09_group_collectives 10_error_handling 

.PHONY: all
all: $(programs)

define build_program =
$(1): $$($(1).cpp)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(1).cpp -o $(1)
endef

$(foreach program,$(programs),$(eval $(call build_program,$(program))))

clean:
	$(RM) $(programs)

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
