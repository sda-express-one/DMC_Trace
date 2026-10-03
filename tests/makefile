# Regression tests.
#   make -C tests          build every test_*.cpp
#   make -C tests run      build and run them all; exits non-zero if any test fails
#   make -C tests clean
#
# The project sources are compiled separately into build/tests/obj (with header dependency tracking),
# never reusing the main build's objects, and asserts stay enabled: the invariants they guard are part
# of what is being tested. Project headers are included relative to include/, as in the main build;
# include paths for Eigen and simplemc come from CPLUS_INCLUDE_PATH.

ROOT = ..

CXX = mpic++
CXXFLAGS = --std=c++23 -O2 -g -Wall -Wextra -Werror -Wpedantic -MMD -MP -I$(ROOT)/include
LDFLAGS = -L/home/samuele/.local/lib -Wl,-rpath=/home/samuele/.local/lib -Wl,--start-group -lsimplemc-mpi -lsimplemc-numeric -lsimplemc-serialize-json -lsimplemc-utils -Wl,--end-group -L/usr/local/lib -lfmt

BUILD_DIR = $(ROOT)/build/tests

SOURCES = $(shell find $(ROOT)/src -name '*.cpp')
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
