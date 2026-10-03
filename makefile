# default to release mode
BUILD ?= release

CXX = mpic++
# project headers are included relative to include/ (e.g. "diagram/vertex.hpp") from every directory
CXXFLAGS = --std=c++23 -Wall -Wextra -Werror -Wpedantic -I$(INCLUDE_DIR)
LDFLAGS = -L/home/samuele/.local/lib -Wl,-rpath=/home/samuele/.local/lib -Wl,--start-group -lsimplemc-mpi -lsimplemc-numeric -lsimplemc-serialize-json -lsimplemc-utils -Wl,--end-group -L/usr/local/lib -lfmt

# directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build/$(BUILD)
BIN_DIR = bin

# target name
PROGRAM_NAME = diagramc

# sources files and corresponding object files - recursive, since sources now live under
# per-module subdirectories (src/diagram, src/updates, ...) rather than directly in src/
SOURCES = $(shell find $(SRC_DIR) -name '*.cpp')

# map src/file.cpp -> build/release/foo.o (preserving directory structure)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# build-specific flags
DEBUG_FLAGS = -g -O0 -fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG

# select flags based on BUILD variable
ifeq ($(BUILD), debug)
	CXXFLAGS += $(DEBUG_FLAGS)
else
	CXXFLAGS += $(RELEASE_FLAGS)
endif

# final target with build directory
TARGET = $(BIN_DIR)/$(PROGRAM_NAME)

# default target
all: $(TARGET)

# link to the program
$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# compile source files (preserving subdirectory structure in build)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# order-only prerequisites above need actual rules to create these directories - mkdir -p in
# the compile recipe above also handles per-file subdirectories (e.g. build/release/diagram/),
# but $(BUILD_DIR)/$(BIN_DIR) themselves still need to exist first for that mkdir invocation.
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# convenience targets
debug:
	$(MAKE) BUILD=debug

release:
	$(MAKE) BUILD=release

clean:
	rm -f $(OBJECTS)

cleanall:
	rm -rf build/ $(BIN_DIR)

.PHONY: all debug release clean cleanall
