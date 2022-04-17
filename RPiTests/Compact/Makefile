GPP := g++-10 -std=c++17
build := ./build_files

rpi_tests_exe := $(build)/rpi_tests


# directories
libimage := ../../libimage_compact
stb := $(libimage)/stb_wrapper


# main
main_c         := rpi_compact_tests.cpp
main_o         := $(build)/rpi_tests_main.o
object_files   := $(main_o)


# libimage
libimage_h += $(libimage)/defines.hpp
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


# flags
INCLUDES  :=
LIBRARIES :=
#CCFLAGS := -march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard

GPP += -O3


# build rules

$(main_o): $(main_c) $(libimage_h)
	@echo "\n main"
	$(GPP) -o $@ -c $<

$(stb_libimage_o): $(stb_libimage_c) $(libimage_h) $(stb_h)
	@echo "\n stb_libimage"
	$(GPP) -o $@ -c $<

$(libimage_o): $(libimage_c) $(libimage_h)
	@echo "\n libimage"
	$(GPP) -o $@ -c $<

$(rpi_tests_exe): $(object_files)
	@echo "\n rpi_tests"
	$(GPP) -o $@ $+ $(LIBRARIES)


build: $(rpi_tests_exe)

run: build
	$(rpi_tests_exe)

clean:
	rm -rfv $(build)/*

setup:
	mkdir -p $(build)