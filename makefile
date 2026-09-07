# default to release mode
BUILD ?= release

CXX = mpic++
CXXFLAGS = --std=c++23 -Wall -Wextra -Werror -Wpedantic
LDFLAGS = -L/home/samuele/.local/lib -Wl,-rpath=/home/samuele/.local/lib -Wl,--start-group -lsimplemc-mpi -lsimplemc-numeric -lsimplemc-serialize-json -lsimplemc-utils -Wl,--end-group -I/usr/local/include -L/usr/local/include -lfmt

# directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build/$(BUILD)
BIN_DIR = bin

# target name
PROGRAM_NAME = diagramc

# sources files and corresponding object files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
# SOURCES += $(wildcard $(SRC_DIR))

# map src/file.cpp -> build/release/foo.o (preserving directory structure)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)%.o,$(SOURCES))

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
