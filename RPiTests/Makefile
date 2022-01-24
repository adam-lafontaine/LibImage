GPP := g++-10 -std=c++17
build := ./build_files

rpi_tests_exe := $(build)/rpi_tests

# directories
libimage := ../libimage
math := $(libimage)/math
proc := $(libimage)/proc
stb := $(libimage)/stb_wrapper
simd := $(proc)/simd


# main
main_c         := rpi_tests_main.cpp
main_o         := $(build)/rpi_tests_main.o
object_files   := $(main_o)


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
math_h := $(math)/math.hpp
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
proc_h += $(proc)/binarize.hpp
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

binarize_c     := $(proc)/binarize.cpp
binarize_o     := $(build)/binarize.o
object_files   += $(binarize_o)

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


# flags
INCLUDES  :=
LIBRARIES :=
CCFLAGS := -march=armv8-a -mtune=cortex-a72 -mfpu=neon-fp-armv8 -mfloat-abi=hard

GPP += -O3


# build rules

$(main_o): $(main_c) $(libimage_h) $(proc_h) $(math_h) $(charts_h)
	@echo "\n main"
	$(GPP) -o $@ -c $<

$(stb_libimage_o): $(stb_libimage_c) $(libimage_h) $(stb_h)
	@echo "\n stb_libimage"
	$(GPP) -o $@ -c $<

$(libimage_o): $(libimage_c) $(libimage_h)
	@echo "\n libimage"
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

$(binarize_o): $(binarize_c) $(libimage_h) $(proc_h)
	@echo "\n binarize"
	$(GPP) -o $@ -c $<

$(blur_o): $(blur_c) $(libimage_h) $(proc_h)
	@echo "\n blur"
	$(GPP) -o $@ -c $<

$(contrast_o): $(contrast_c) $(libimage_h) $(proc_h)
	@echo "\n contrast"
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

$(simd_alpha_blend_o): $(simd_alpha_blend_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_alpha_blend"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(simd_convolve_o): $(simd_convolve_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_convolve"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(simd_grayscale_o): $(simd_grayscale_c) $(libimage_h) $(proc_h) $(simd_h)
	@echo "\n simd_grayscale"
	$(GPP) $(CCFLAGS) -o $@ -c $<

$(rpi_tests_exe): $(object_files)
	@echo "\n rpi_tests"
	$(GPP) -o $@ $+ $(LIBRARIES)


build: $(rpi_tests_exe)

run: build
	$(rpi_tests_exe)

clean:
	rm -rfv $(build)/*
	rm -rfv ./out_files/*

setup:
	mkdir -p $(build)