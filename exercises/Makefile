CXX := icpx
CXXFLAGS := -O2 -std=c++17
SYCLFLAGS := -fsycl
CXXFLAGS += $(SYCLFLAGS)

programs = 01_more_device_info 

.PHONY: all
all: $(programs)

define build_program =
$(1): $$($(1).cpp)
	$(CXX) $(CXXFLAGS) $(1).cpp -o $(1)
endef

$(foreach program,$(programs),$(eval $(call build_program,$(program))))

clean:
	$(RM) $(programs)

print-info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"