# Regression tests.
#   make -C tests          build every test_*.cpp
#   make -C tests run      build and run them all; exits non-zero if any test fails
#   make -C tests clean
#
# The project sources are compiled separately into build/tests/obj (with header dependency tracking),
# never reusing the main build's objects, and asserts stay enabled: the invariants they guard are part
# of what is being tested. Project headers are included relative to include/, as in the main build;
# the external paths (simplemc, Eigen, nlohmann/json, fmt) and the compiler come from ../config.mk.

ROOT = ..
include $(ROOT)/config.mk

# DMC_ROOT: absolute path of the repository, for tests that read files from it (test_run_input: inputs/)
CXXFLAGS = --std=c++23 -O2 -g -Wall -Wextra -Werror -Wpedantic -MMD -MP -I$(ROOT)/include $(EXTERNAL_INCLUDES) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
           -DDMC_ROOT=\"$(abspath $(ROOT))\"
LDFLAGS = $(EXTERNAL_LIBS)

BUILD_DIR = $(ROOT)/build/tests

# src/main.cpp is the program's entry point: the test executables have their own main
SOURCES = $(filter-out $(ROOT)/src/main.cpp,$(shell find $(ROOT)/src -name '*.cpp'))
OBJECTS = $(patsubst $(ROOT)/src/%.cpp,$(BUILD_DIR)/obj/%.o,$(SOURCES))

TESTS = $(basename $(wildcard test_*.cpp))
BINARIES = $(addprefix $(BUILD_DIR)/,$(TESTS))

all: $(BINARIES)

$(BUILD_DIR)/test_%: test_%.cpp $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $< $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/obj/%.o: $(ROOT)/src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: all
	@failed=""; \
	for t in $(TESTS); do \
		$(BUILD_DIR)/$$t || failed="$$failed $$t"; \
	done; \
	if [ -n "$$failed" ]; then echo "FAILED:$$failed"; exit 1; else echo "all $(words $(TESTS)) tests passed"; fi

clean:
	rm -rf $(BUILD_DIR)

-include $(OBJECTS:.o=.d) $(BINARIES:=.d)

# the objects are only reached through the pattern rules above, so make would otherwise treat them as
# intermediate files and delete them after every build
.SECONDARY: $(OBJECTS)

.PHONY: all run clean
