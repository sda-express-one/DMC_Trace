# Machine-specific build paths, included by makefile and tests/makefile.
#
# Defaults are those of the development laptop. On another machine either
#   - put the overrides in config.local.mk next to this file (git-ignored, read first), e.g.
#         SIMPLEMC_PREFIX = /opt/simplemc
#         EIGEN_INC       = $(EBROOTEIGEN)/include
#   - or pass them on the command line: make SIMPLEMC_PREFIX=/opt/simplemc
# Every variable here is set with ?=, so either way wins over these defaults.
#
# The external headers are added with -isystem: no warnings from them under -Werror, and -MMD does not track
# them (rebuild with make clean after updating a library).

-include $(dir $(lastword $(MAKEFILE_LIST)))config.local.mk

# MPI compiler wrapper (OpenMPI or MPICH; GCC >= 13 underneath for C++23). make predefines CXX = g++, so ?= would
# never apply: replace only that built-in default, keep any value from the command line, environment or local file.
ifeq ($(origin CXX),default)
    CXX = mpic++
endif

# simplemc install prefix: headers in include/simplemc, libsimplemc-*.a in lib
SIMPLEMC_PREFIX ?= $(HOME)/.local
# Eigen 3 headers (the directory containing Eigen/)
EIGEN_INC ?= /usr/include/eigen3
# nlohmann/json headers (the directory containing nlohmann/); empty: the compiler's default search path
JSON_INC ?=
# directory of libfmt.a / libfmt.so; empty: the linker's default search path
FMT_LIB ?= /usr/local/lib

EXTERNAL_INCLUDES = -isystem $(SIMPLEMC_PREFIX)/include -isystem $(EIGEN_INC) $(if $(JSON_INC),-isystem $(JSON_INC))
EXTERNAL_LIBS = -L$(SIMPLEMC_PREFIX)/lib -Wl,-rpath=$(SIMPLEMC_PREFIX)/lib \
                -Wl,--start-group -lsimplemc-mpi -lsimplemc-numeric -lsimplemc-serialize-json -lsimplemc-utils -Wl,--end-group \
                $(if $(FMT_LIB),-L$(FMT_LIB)) -lfmt
