# Machine-specific build paths, included by makefile and tests/makefile.
#
# Defaults are those of the development laptop. On another machine either
#   - put the overrides in config.local.mk next to this file (git-ignored, read first), e.g.
#         SIMPLEMC_PREFIX = /opt/simplemc
#         SIMPLEMC_LIB    = /opt/simplemc/lib64      (only if the automatic lib/lib64 choice is wrong)
#         EIGEN_INC       = $(EBROOTEIGEN)/include
#         FMT_LIB         = $(EBROOTFMT)/lib64
#         MPICXX          = mpiicpc                  (only if mpicxx is not the wrapper you want)
#   - or pass them on the command line: make SIMPLEMC_PREFIX=/opt/simplemc
# Every variable here is set with ?=, so either way wins over these defaults.
#
# The external headers are added with -isystem: no warnings from them under -Werror, and -MMD does not track
# them (rebuild with make clean after updating a library).

-include $(dir $(lastword $(MAKEFILE_LIST)))config.local.mk

# MPI C++ compiler wrapper: mpicxx exists in OpenMPI, MPICH and Intel MPI (GCC >= 13 underneath for C++23). The build
# always compiles and links with it: CXX is set from MPICXX, not taken from the environment, because cluster compiler
# modules commonly export CXX as the serial compiler (g++, icpx), which compiles (mpi.h is on CPATH) but then fails to
# link the MPI library. Override MPICXX, not CXX (e.g. MPICXX = mpiicpc).
MPICXX ?= mpicxx
CXX = $(MPICXX)

# simplemc install prefix: headers in include/simplemc
SIMPLEMC_PREFIX ?= $(HOME)/.local
# directory of libsimplemc-*.a: CMake installs to lib or lib64 depending on the system (GNUInstallDirs); found
# automatically (lib64 if the libraries are there, else lib), or set it explicitly
SIMPLEMC_LIB ?= $(if $(wildcard $(SIMPLEMC_PREFIX)/lib64/libsimplemc-utils.a),$(SIMPLEMC_PREFIX)/lib64,$(SIMPLEMC_PREFIX)/lib)
# Eigen 3 headers (the directory containing Eigen/)
EIGEN_INC ?= /usr/include/eigen3
# nlohmann/json headers (the directory containing nlohmann/); empty: the compiler's default search path
JSON_INC ?=
# directory of libfmt.a / libfmt.so; empty: the linker's default search path
FMT_LIB ?= /usr/local/lib

EXTERNAL_INCLUDES = -isystem $(SIMPLEMC_PREFIX)/include -isystem $(EIGEN_INC) $(if $(JSON_INC),-isystem $(JSON_INC))
EXTERNAL_LIBS = -L$(SIMPLEMC_LIB) -Wl,-rpath=$(SIMPLEMC_LIB) \
                -Wl,--start-group -lsimplemc-mpi -lsimplemc-numeric -lsimplemc-serialize-json -lsimplemc-utils -Wl,--end-group \
                $(if $(FMT_LIB),-L$(FMT_LIB)) -lfmt
