################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda-10.2

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

# When on native aarch64 system with userspace of 32-bit, change TARGET_ARCH to armv7l
ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_SIZE),aarch64-aarch64-32)
    TARGET_ARCH = armv7l
endif

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
ifeq (,$(filter $(TARGET_OS),linux darwin qnx android))
    $(error ERROR - unsupported value $(TARGET_OS) for TARGET_OS!)
endif

# host compiler
ifeq ($(TARGET_OS),darwin)
    ifeq ($(shell expr `xcodebuild -version | grep -i xcode | awk '{print $$2}' | cut -d'.' -f1` \>= 5),1)
        HOST_COMPILER ?= clang++
    endif
else ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        ifeq ($(TARGET_OS),linux)
            HOST_COMPILER ?= arm-linux-gnueabihf-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/arm-unknown-nto-qnx6.6.0eabi-g++
        else ifeq ($(TARGET_OS),android)
            HOST_COMPILER ?= arm-linux-androideabi-g++
        endif
    else ifeq ($(TARGET_ARCH),aarch64)
        ifeq ($(TARGET_OS), linux)
            HOST_COMPILER ?= aarch64-linux-gnu-g++
        else ifeq ($(TARGET_OS),qnx)
            ifeq ($(QNX_HOST),)
                $(error ERROR - QNX_HOST must be passed to the QNX host toolchain)
            endif
            ifeq ($(QNX_TARGET),)
                $(error ERROR - QNX_TARGET must be passed to the QNX target toolchain)
            endif
            export QNX_HOST
            export QNX_TARGET
            HOST_COMPILER ?= $(QNX_HOST)/usr/bin/q++
        else ifeq ($(TARGET_OS), android)
            HOST_COMPILER ?= aarch64-linux-android-clang++
        endif
    else ifeq ($(TARGET_ARCH),ppc64le)
        HOST_COMPILER ?= powerpc64le-linux-gnu-g++
    endif
endif
HOST_COMPILER ?= g++
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=

# build flags
ifeq ($(TARGET_OS),darwin)
    LDFLAGS += -rpath $(CUDA_PATH)/lib
    CCFLAGS += -arch $(HOST_ARCH)
else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
    LDFLAGS += --dynamic-linker=/lib/ld-linux-armhf.so.3
    CCFLAGS += -mfloat-abi=hard
else ifeq ($(TARGET_OS),android)
    LDFLAGS += -pie
    CCFLAGS += -fpie -fpic -fexceptions
endif

ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
        ifneq ($(TARGET_FS),)
            GCCVERSIONLTEQ46 := $(shell expr `$(HOST_COMPILER) -dumpversion` \<= 4.6)
            ifeq ($(GCCVERSIONLTEQ46),1)
                CCFLAGS += --sysroot=$(TARGET_FS)
            endif
            LDFLAGS += --sysroot=$(TARGET_FS)
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib -L$(TARGET_FS)/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/lib/aarch64-linux-gnu -L$(TARGET_FS)/lib/aarch64-linux-gnu
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib -L$(TARGET_FS)/usr/lib
            LDFLAGS += -rpath-link=$(TARGET_FS)/usr/lib/aarch64-linux-gnu -L$(TARGET_FS)/usr/lib/aarch64-linux-gnu
            LDFLAGS += --unresolved-symbols=ignore-in-shared-libs
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include  -I$(TARGET_FS)/usr/include
            CCFLAGS += -isystem=$(TARGET_FS)/usr/include/aarch64-linux-gnu -I$(TARGET_FS)/usr/include/aarch64-linux-gnu
        endif
    endif
    ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
        NVCCFLAGS += --qpp-config 5.4.0,gcc_ntoaarch64le
        CCFLAGS += -DWIN_INTERFACE_CUSTOM -I/usr/include/aarch64-qnx-gnu
        LDFLAGS += -lsocket
        LDFLAGS += -L/usr/lib/aarch64-qnx-gnu
        CCFLAGS += "-Wl\,-rpath-link\,/usr/lib/aarch64-qnx-gnu"
        ifdef TARGET_OVERRIDE
            LDFLAGS += -lslog2
        endif

        ifneq ($(TARGET_FS),)
            LDFLAGS += -L$(TARGET_FS)/usr/lib
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/lib"
            LDFLAGS += -L$(TARGET_FS)/usr/libnvidia
            CCFLAGS += "-Wl\,-rpath-link\,$(TARGET_FS)/usr/libnvidia"
            CCFLAGS += -I$(TARGET_FS)/../include
        endif
    endif
endif

ifdef TARGET_OVERRIDE # cuda toolkit targets override
    NVCCFLAGS += -target-dir $(TARGET_OVERRIDE)
endif

# Install directory of different arch
CUDA_INSTALL_TARGET_DIR :=
ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-linux)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-gnueabihf/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-linux)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-android)
    CUDA_INSTALL_TARGET_DIR = targets/armv7-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-android)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-linux-androideabi/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),armv7l-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/ARMv7-linux-QNX/
else ifeq ($(TARGET_ARCH)-$(TARGET_OS),aarch64-qnx)
    CUDA_INSTALL_TARGET_DIR = targets/aarch64-qnx/
else ifeq ($(TARGET_ARCH),ppc64le)
    CUDA_INSTALL_TARGET_DIR = targets/ppc64le-linux/
endif

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
#INCLUDES  := -I../../common/inc
#INCLUDES  := -I../samples/common/inc
INCLUDES   :=
LIBRARIES :=

################################################################################

# Gencode arguments
ifeq ($(TARGET_ARCH),$(filter $(TARGET_ARCH),armv7l aarch64))
SMS ?= 30 32 53 61 62 70 72 75
else
SMS ?= 30 61 70 75
endif

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif

################################################################################

GPP := g++-10 -std=c++17
build := ./build_files

cuda_test_exe := $(build)/cuda_test

# directories
libimage := ../libimage
math := $(libimage)/math
proc := $(libimage)/proc
stb := $(libimage)/stb_wrapper
cuda := $(libimage)/cuda
simd := $(proc)/simd


# main
main_c         := cuda_test_main.cpp
main_o         := $(build)/cuda_test_main.o
object_files   := $(main_o)

types_h := $(libimage)/types.hpp


# libimage
libimage_h := $(libimage)/types.hpp
libimage_h += $(libimage)/defines.hpp
libimage_h += $(libimage)/rgba.hpp
libimage_h += $(libimage)/gray.hpp
libimage_h += $(libimage)/libimage.hpp

libimage_c     := $(libimage)/libimage.cpp
libimage_o     := $(build)/libimage.o
object_files   += $(libimage_o)


# stb
stb_h := $(stb)/stb_image_resize.h
stb_h += $(stb)/stb_image_write.h
stb_h += $(stb)/stb_image.h
stb_h += $(stb)/stb_include.hpp

stb_libimage_c := $(stb)/stb_libimage.cpp
stb_libimage_o := $(build)/stb_libimage.o
object_files   += $(stb_libimage_o)


#math/charts
math_h   := $(math)/math.hpp
charts_h := $(math)/charts.hpp

math_c         := $(math)/math.cpp
math_o         := $(build)/math.o
object_files   += $(math_o)

charts_c       := $(math)/charts.cpp
charts_o       := $(build)/charts.o
object_files   += $(charts_o)


# image processing
proc_h := $(proc)/process.hpp
proc_h += $(proc)/alpha_blend.hpp
proc_h += $(proc)/binary.hpp
proc_h += $(proc)/blur.hpp
proc_h += $(proc)/contrast.hpp
proc_h += $(proc)/convolve.hpp
proc_h += $(proc)/copy.hpp
proc_h += $(proc)/edges.hpp
proc_h += $(proc)/gradients.hpp
proc_h += $(proc)/grayscale.hpp
proc_h += $(proc)/transform.hpp
proc_h += $(proc)/verify.hpp

proc_h += $(proc)/index_range.hpp # TODO: utils

alpha_blend_c  := $(proc)/alpha_blend.cpp
alpha_blend_o  := $(build)/alpha_blend.o
object_files   += $(alpha_blend_o)

binary_c     := $(proc)/binary.cpp
binary_o     := $(build)/binary.o
object_files   += $(binary_o)

blur_c         := $(proc)/blur.cpp
blur_o         := $(build)/blur.o
object_files   += $(blur_o)

contrast_c     := $(proc)/contrast.cpp
contrast_o     := $(build)/contrast.o
object_files   += $(contrast_o)

convolve_c     := $(proc)/convolve.cpp
convolve_o     := $(build)/convolve.o
object_files   += $(convolve_o)

copy_c         := $(proc)/copy.cpp
copy_o         := $(build)/copy.o
object_files   += $(copy_o)

edges_gradients_c  := $(proc)/edges_gradients.cpp
edges_gradients_o  := $(build)/edges_gradients.o
object_files       += $(edges_gradients_o)

grayscale_c    := $(proc)/grayscale.cpp
grayscale_o    := $(build)/grayscale.o
object_files   += $(grayscale_o)

transform_c    := $(proc)/transform.cpp
transform_o    := $(build)/transform.o
object_files   += $(transform_o)

rotate_c    := $(proc)/rotate.cpp
rotate_o    := $(build)/rotate.o
object_files   += $(rotate_o)


# SIMD
simd_h := $(simd)/simd_def.hpp

simd_alpha_blend_c  := $(simd)/simd_alpha_blend.cpp
simd_alpha_blend_o  := $(build)/simd_alpha_blend.o
object_files        += $(simd_alpha_blend_o)

simd_convolve_c := $(simd)/simd_convolve.cpp
simd_convolve_o := $(build)/simd_convolve.o
object_files    += $(simd_convolve_o)

simd_grayscale_c := $(simd)/simd_grayscale.cpp
simd_grayscale_o := $(build)/simd_grayscale.o
object_files     += $(simd_grayscale_o)


# CUDA
gpu_h := $(types_h)
gpu_h += $(libimage)/rgba.hpp
gpu_h += $(libimage)/gray.hpp
gpu_h += $(cuda)/cuda_def.cuh
gpu_h += $(cuda)/convolve.cuh
gpu_h += $(cuda)/device.hpp
gpu_h += $(cuda)/device_image.hpp
gpu_h += $(cuda)/process.hpp

gpu_device_c   := $(cuda)/device.cu
gpu_device_o   := $(build)/gpu_device.o
object_files   += $(gpu_device_o)

gpu_device_image_c   := $(cuda)/device_image.cpp
gpu_device_image_o   := $(build)/gpu_device_image.o
object_files         += $(gpu_device_image_o)

gpu_process_c   := $(cuda)/process.cu
gpu_process_o   := $(build)/gpu_process.o
object_files    += $(gpu_process_o)

# flags
CCFLAGS := 

#GPP += -O3


check.deps: 
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

$(main_o): $(main_c) $(libimage_h) $(proc_h) $(math_h) $(charts_h)
	@echo "\n main"
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(libimage_o): $(libimage_c) $(libimage_h) $(libimage_h) $(stb_h)
	@echo "\n libimage"
	$(GPP) -o $@ -c $<

$(stb_libimage_o): $(stb_libimage_c) $(libimage_h)
	@echo "\n stb_libimage"
	$(GPP) -o $@ -c $<

$(math_o): $(math_c) $(libimage_h) $(math_h)
	@echo "\n math"
	$(GPP) -o $@ -c $<

$(charts_o): $(charts_c) $(libimage_h) $(charts_h)
	@echo "\n charts"
	$(GPP) -o $@ -c $<

$(alpha_blend_o): $(alpha_blend_c) $(libimage_h) $(proc_h)
	@echo "\n alpha_blend"
	$(GPP) -o $@ -c $<

$(binary_o): $(binary_c) $(libimage_h) $(proc_h)
	@echo "\n binary"
	$(GPP) -o $@ -c $<

$(blur_o): $(blur_c) $(libimage_h) $(proc_h)
	@echo "\n blur"
	$(GPP) -o $@ -c $<

$(contrast_o): $(contrast_c) $(libimage_h) $(proc_h)
	@echo "\n contrast.o"
	$(GPP) -o $@ -c $<

$(convolve_o): $(convolve_c) $(libimage_h) $(proc_h)
	@echo "\n convolve"
	$(GPP) -o $@ -c $<

$(copy_o): $(copy_c) $(libimage_h) $(proc_h)
	@echo "\n copy"
	$(GPP) -o $@ -c $<

$(edges_gradients_o): $(edges_gradients_c) $(libimage_h) $(proc_h)
	@echo "\n edges_gradients"
	$(GPP) -o $@ -c $<

$(grayscale_o): $(grayscale_c) $(libimage_h) $(proc_h)
	@echo "\n grayscale"
	$(GPP) -o $@ -c $<

$(transform_o): $(transform_c) $(libimage_h) $(proc_h)
	@echo "\n transform"
	$(GPP) -o $@ -c $<

$(rotate_o): $(rotate_c) $(libimage_h) $(proc_h)
	@echo "\n rotate"
	$(GPP) -o $@ -c $<

$(simd_alpha_blend_o): $(simd_alpha_blend_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_alpha_blend"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(simd_convolve_o): $(simd_convolve_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_convolve"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(simd_grayscale_o): $(simd_grayscale_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_grayscale"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(gpu_device_o): $(gpu_device_c) $(gpu_h)
	@echo "\n gpu_device"
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(gpu_device_image_o): $(gpu_device_image_c) $(gpu_h)
	@echo "\n gpu_device_image"
	$(GPP) -o $@ -c $<

$(gpu_process_o): $(gpu_process_c) $(gpu_h)
	@echo "\n gpu_process"
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(cuda_test_exe): $(object_files)
	@echo "\n cuda_test"
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
    

build: $(cuda_test_exe)

run: build
	$(EXEC) $(cuda_test_exe)

clean:
	rm -rfv $(build)/*
	rm -rfv ./out_files/*

setup:
	mkdir -p $(build)