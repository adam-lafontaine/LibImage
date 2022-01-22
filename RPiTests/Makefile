GPP := g++-10 -std=c++17
build := ./build_files

rpi_tests_exe := $(build)/rpi_tests

# directories
libimage := ../libimage
math := $(libimage)/math
proc := $(libimage)/proc
stb := $(libimage)/stb_wrapper


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


# build rules

INCLUDES  :=
LIBRARIES :=

GPP += -O3


$(main_o): $(main_c) $(libimage_h) $(proc_h) $(math_h) $(charts_h)
	@echo "\n main"
	$(GPP) -o $@ -c $<

$(stb_libimage_o): $(stb_libimage_c) $(libimage_h) $(stb_h)
	@echo "\n stb_libimage.o"
	$(GPP) -o $@ -c $<

$(libimage_o): $(libimage_c) $(libimage_h)
	@echo "\n libimage.o"
	$(GPP) -o $@ -c $<

$(math_o): $(math_c) $(libimage_h) $(math_h)
	@echo "\n math.o"
	$(GPP) -o $@ -c $<

$(charts_o): $(charts_c) $(libimage_h) $(charts_h)
	@echo "\n charts.o"
	$(GPP) -o $@ -c $<

$(alpha_blend_o): $(alpha_blend_c) $(libimage_h) $(proc_h)
	@echo "\n alpha_blend.o"
	$(GPP) -o $@ -c $<

$(binarize_o): $(binarize_c) $(libimage_h) $(proc_h)
	@echo "\n binarize.o"
	$(GPP) -o $@ -c $<

$(blur_o): $(blur_c) $(libimage_h) $(proc_h)
	@echo "\n blur.o"
	$(GPP) -o $@ -c $<

$(contrast_o): $(contrast_c) $(libimage_h) $(proc_h)
	@echo "\n contrast.o"
	$(GPP) -o $@ -c $<

$(convolve_o): $(convolve_c) $(libimage_h) $(proc_h)
	@echo "\n convolve.o"
	$(GPP) -o $@ -c $<

$(copy_o): $(copy_c) $(libimage_h) $(proc_h)
	@echo "\n copy.o"
	$(GPP) -o $@ -c $<

$(edges_gradients_o): $(edges_gradients_c) $(libimage_h) $(proc_h)
	@echo "\n edges_gradients.o"
	$(GPP) -o $@ -c $<

$(grayscale_o): $(grayscale_c) $(libimage_h) $(proc_h)
	@echo "\n grayscale.o"
	$(GPP) -o $@ -c $<

$(transform_o): $(transform_c) $(libimage_h) $(proc_h)
	@echo "\n transform.o"
	$(GPP) -o $@ -c $<

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