# CURAFFT Makefile

CC   = gcc
CXX  = g++
NVCC = nvcc

#set based on GPU card, sm_60 (Tesla P100) or sm_61 (consumer Pascal) or sm_70 (Tesla V100, Titan V) or sm_80 (A100)
NVARCH ?= -gencode=arch=compute_70,code=sm_70



CFLAGS    ?= -fPIC -O3 -funroll-loops -march=native
CXXFLAGS  ?= $(CFLAGS) -std=c++14
NVCCFLAGS ?= -std=c++14 -ccbin=$(CXX) -O3 $(NVARCH) -Wno-deprecated-gpu-targets \
	     --default-stream per-thread -Xcompiler "$(CXXFLAGS)"


#NVCCFLAGS+= -DINFO -DDEBUG -DRESULT -DTIME
#NVCCFLAGS+= -DDEBUG

#set your cuda path
CUDA_ROOT := /usr/local/cuda

# Common includes
INC += -I$(CUDA_ROOT)/include -Iinclude/cuda_sample

# libs
NVCC_LIBS_PATH += -L$(CUDA_ROOT)/lib64

ifdef NVCC_STUBS
    $(info detected CUDA_STUBS -- setting CUDA stubs directory)
    NVCC_LIBS_PATH += -L$(NVCC_STUBS)
endif

LIBS += -lm -lcudart -lstdc++ -lnvToolsExt -lcufft -lcuda



# Include header files
INC += -I include


LIBNAME=libcurafft
DYNAMICLIB=lib/$(LIBNAME).so
STATICLIB=lib-static/$(LIBNAME).a

BINDIR=bin

HEADERS = include/curafft_opts.h include/curafft_plan.h include/cugridder.h \
	include/conv_interp_invoker.h include/conv.h include/interp.h include/cuft.h include/dataType.h \
	include/deconv.h include/precomp.h include/ragridder_plan.h include/utils.h \
	contrib/common.h contrib/legendre_rule_fast.h contrib/utils_fp.h
# later put some file into the contrib
CONTRIBOBJS=contrib/common.o contrib/utils_fp.o

CURAFFTOBJS=src/utils.o contrib/legendre_rule_fast.o

CURAFFTOBJS_64=src/FT/conv_interp_invoker.o src/FT/conv.o src/FT/interp.o src/FT/cuft.o src/FT/deconv.o \
	src/RA/cugridder.o src/RA/precomp.o src/RA/ra_exec.o $(CONTRIBOBJS)

#ignore single precision first



%.o: %.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@
%.o: %.c $(HEADERS)
	$(CC) -c $(CFLAGS) $(INC) $< -o $@
%.o: %.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

src/%.o: src/%.cpp $(HEADERS)
	$(CXX) -c $(CXXFLAGS) $(INC) $< -o $@

src/%.o: src/%.c $(HEADERS)
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

src/%.o: src/%.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

src/FT/%.o: src/FT/%.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

src/RA/%.o: src/FT/%.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

test/%.o: test/%.cu $(HEADERS)
	$(NVCC) --device-c -c $(NVCCFLAGS) $(INC) $< -o $@

default: all


all: libtest explicit_gridder_test checkadjoint

# testers for the lib (does not execute)
libtest: lib convtest utiltest w_s_test nufft_test

convtest: $(BINDIR)/conv_2d_test \
	$(BINDIR)/conv_3d_test

explicit_gridder_test: $(BINDIR)/explicit_gridder_test

utiltest: $(BINDIR)/utils_test

w_s_test: $(BINDIR)/w_s_gridder_test \
	$(BINDIR)/w_s_degridder_test

nufft_test: $(BINDIR)/nufft_1d_test \
	$(BINDIR)/nufft_2d_1_test \
	$(BINDIR)/nufft_2d_2_test \
	$(BINDIR)/nufft_1d_3_1_test \
	$(BINDIR)/nufft_1d_3_2_test

adjointness_test: $(BINDIR)/adjointness_1d_test


$(BINDIR)/%: test/%.o $(CURAFFTOBJS_64) $(CURAFFTOBJS)
	mkdir -p $(BINDIR)
	$(NVCC) $^ $(NVCCFLAGS) $(NVCC_LIBS_PATH) $(LIBS) -o $@



# user-facing library...
lib: $(STATICLIB) $(DYNAMICLIB)
# add $(CONTRIBOBJS) to static and dynamic later
$(STATICLIB): $(CURAFFTOBJS) $(CURAFFTOBJS_64) $(CONTRIBOBJS)
	mkdir -p lib-static
	ar rcs $(STATICLIB) $^
$(DYNAMICLIB): $(CURAFFTOBJS) $(CURAFFTOBJS_64) $(CONTRIBOBJS)
	mkdir -p lib
	$(NVCC) -shared $(NVCCFLAGS) $^ -o $(DYNAMICLIB) $(LIBS)


# ---------------------------------------------------------------
check:
	@echo "Building lib, all testers, and running all tests..."
	$(MAKE) checkconv


checkconv: libtest convtest
	@echo "Running conv/interp only tests..."
	@echo "conv 2D.............................................."
	bin/conv_2d_test 0 5 5
	@echo "conv 3D.............................................."
	bin/conv_3d_test 0 5 5 2

checkutils: utiltest
	@echo "Utilities checking..."
	bin/utils_test

checkwst: w_s_test

	@echo "W stacking checking..."
	bin/w_s_gridder_test 0 1 100 100 10000 10
	bin/w_s_degridder_test 0 1 100 100 10000 10
# bin/w_s_test 0 1 5000 5000 50000000 10

checkeg: explicit_gridder_test
	@echo "Explicit gridder testing..."
	bin/explicit_gridder_test 20 20 20 10

checkfft: nufft_test
	@echo "NUFFT testing..."
	bin/nufft_1d_test 4096 4096 1e-13
	bin/nufft_2d_1_test 10 10 100 1e-12
	bin/nufft_2d_2_test 10 10 100 1e-12
	bin/nufft_2d_2_test 100 100 10000 1e-12
	bin/nufft_1d_3_1_test
	bin/nufft_1d_3_2_test


checkadjoint: adjointness_test
	@echo "adjointness testing..."
	bin/adjointness_1d_test


python: libtest
	cp lib/libcurafft.so python/curagridder/
# -----------------------------------------------------------------

clean:
	rm -f *.o
	rm -f test/*.o
	rm -f src/*.o
	rm -f src/FT/*.o
	rm -f src/RA/*.o
	rm -f contrib/*.o
	rm -rf $(BINDIR)
	rm -rf lib
	rm -rf lib-static

.PHONY: default all libtest convtest check checkconv
.PHONY: clean
