CXX := clang++
CXXFLAGS := -O2 -std=c++17
SYCLFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

programs = 01_more_device_info 02_device_selection 03_batch_axpy 04_kernel_fusion

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
