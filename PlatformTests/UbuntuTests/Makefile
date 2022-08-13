GPP := g++-10 -std=c++17
build := ./build_files

ubuntu_tests_exe := $(build)/ubuntu_tests


# directories
libimage := ../../libimage_parallel
stb := $(libimage)/stb_wrapper


# main
main_c         := ubuntu_parallel_tests.cpp
main_o         := $(build)/ubuntu_tests_main.o
object_files   := $(main_o)

defines_h := $(libimage)/defines.hpp
simd_def_h := $(libimage)/simd_def.hpp


# libimage
libimage_h := $(libimage)/libimage.hpp
libimage_h += $(defines_h)


libimage_c     := $(libimage)/libimage.cpp
libimage_o     := $(build)/libimage.o
object_files   += $(libimage_o)


# stb
stb_h := $(stb)/stb_include.hpp
stb_h += $(stb)/stb_image_resize.h
stb_h += $(stb)/stb_image_write.h
stb_h += $(stb)/stb_image.h
stb_h += $(defines_h)


stb_libimage_c := $(stb)/stb_libimage.cpp
stb_libimage_o := $(build)/stb_libimage.o
object_files   += $(stb_libimage_o)


# flags
INCLUDES  :=
LIBRARIES :=
CCFLAGS := 
#CCFLAGS := -march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard
#CCFLAGS := -mfpmath=sse -mmmx -msse -msse2

GPP += -O3


# build rules

$(main_o): $(main_c) $(libimage_h)
	@echo "\n main"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(stb_libimage_o): $(stb_libimage_c) $(libimage_h) $(stb_h)
	@echo "\n stb_libimage"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(libimage_o): $(libimage_c) $(libimage_h) $(simd_def_h)
	@echo "\n libimage"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(ubuntu_tests_exe): $(object_files)
	@echo "\n ubuntu_tests"
	$(GPP) $(CCFLAGS) -o $@ $+ $(LIBRARIES)


build: $(ubuntu_tests_exe)

run: build
	$(ubuntu_tests_exe)

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)