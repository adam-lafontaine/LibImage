#include "../libimage/libimage.hpp"
#include "../libimage/proc/process.hpp"
#include "../libimage/math/libimage_math.hpp"
#include "../libimage/cuda/process.hpp"
#include "./utils/stopwatch.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <array>

namespace img = libimage;
using path_t = std::string;

using Image = img::image_t;
using ImageView = img::view_t;
using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using Pixel = img::pixel_t;

//constexpr auto ROOT_DIR = "~/Repos/LibImage/CudaTests";
constexpr auto ROOT_DIR = "/home/adam/Repos/LibImage/CudaTests/";

const auto ROOT_PATH = std::string(ROOT_DIR);

// make sure these files exist
const auto CORVETTE_PATH = ROOT_PATH + "in_files/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH + "in_files/cadillac.png";

const auto DST_IMAGE_ROOT = ROOT_PATH + "out_files/";

void empty_dir(path_t& dir);
void process_tests(path_t& out_dir);
void cuda_tests(path_t& out_dir);
void print(img::stats_t const& stats);

int main()
{
	auto dst_root = DST_IMAGE_ROOT;

	//auto dst_proc = dst_root + "proc/";
    //process_tests(dst_proc);

	auto dst_cuda = dst_root + "cuda/";
	cuda_tests(dst_cuda);

    printf("\nDone.\n");
}


void process_tests(path_t& out_dir)
{
	std::cout << "process:\n";
	empty_dir(out_dir);

	// get image
	Image corvette_img;
	img::read_image_from_file(CORVETTE_PATH, corvette_img);
	img::write_image(corvette_img, out_dir + "vette.png");

	auto const width = corvette_img.width;
	auto const height = corvette_img.height;	

	// get another image for blending
	// make sure it is the same size
	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	Image caddy_img;
	caddy_img.width = width;
	caddy_img.height = height;
	img::resize_image(caddy_read, caddy_img);
	img::write_image(caddy_img, out_dir + "caddy.png");


	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage dst_gray_img;
	img::make_image(dst_gray_img, width, height);


	// alpha blending
	img::seq::transform_alpha(caddy_img, [](auto const& p) { return 128; });
	img::seq::alpha_blend(caddy_img, corvette_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend.png");

	img::seq::copy(corvette_img, dst_img);
	img::seq::alpha_blend(caddy_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst.png");

	// grayscale
	img::seq::transform_grayscale(corvette_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "convert_grayscale.png");

	// stats
	auto gray_stats = img::calc_stats(dst_gray_img);
	GrayImage gray_stats_img;
	img::draw_histogram(gray_stats.hist, gray_stats_img);
	img::write_image(gray_stats_img, out_dir + "gray_stats.png");
	print(gray_stats);

	// alpha grayscale
	img::seq::transform_alpha_grayscale(corvette_img);
	auto alpha_stats = img::calc_stats(corvette_img, img::Channel::Alpha);
	GrayImage alpha_stats_img;
	img::draw_histogram(alpha_stats.hist, alpha_stats_img);
	img::write_image(alpha_stats_img, out_dir + "alpha_stats.png");
	print(alpha_stats);

	// create a new grayscale source
	GrayImage src_gray_img;
	img::make_image(src_gray_img, width, height);
	img::seq::copy(dst_gray_img, src_gray_img);

	// contrast
	auto shade_min = static_cast<u8>(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = static_cast<u8>(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));
	img::seq::transform_contrast(src_gray_img, dst_gray_img, shade_min, shade_max);
	img::write_image(dst_gray_img, out_dir + "contrast.png");

	// binarize
	auto const is_white = [&](u8 p) { return static_cast<r32>(p) > gray_stats.mean; };
	img::seq::binarize(src_gray_img, dst_gray_img, is_white);
	img::write_image(dst_gray_img, out_dir + "binarize.png");

	//blur
	img::seq::blur(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur.png");	

	// edge detection
	img::seq::edges(src_gray_img, dst_gray_img, 150);
	img::write_image(dst_gray_img, out_dir + "edges.png");

	// gradient
	img::seq::gradients(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "gradient.png");

	// combine transformations in the same image
	// regular grayscale to start
	img::seq::copy(src_gray_img, dst_gray_img);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_img, range);
	auto dst_sub = img::sub_view(dst_gray_img, range);
	img::seq::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::seq::transform_contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::seq::blur(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::seq::gradients(src_sub, dst_sub);

	img::write_image(dst_gray_img, out_dir + "combo.png");


}


void cuda_tests(path_t& out_dir)
{
	std::cout << "cuda:\n";
	empty_dir(out_dir);

	Image corvette_img;
	img::read_image_from_file(CORVETTE_PATH, corvette_img);
	auto width = corvette_img.width;
	auto height = corvette_img.height;

	Image img_read;
	img::read_image_from_file(CADILLAC_PATH, img_read);

	Image caddy_img;
	caddy_img.width = width;
	caddy_img.height = height;
	img::resize_image(img_read, caddy_img);
	

	Image src_img;
	img::make_image(src_img, width, height);

	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage src_gray_img;
	img::make_image(src_gray_img, width, height);

	GrayImage dst_gray_img;
	img::make_image(dst_gray_img, width, height);

	
	// setup device memory for color images
	DeviceBuffer<img::pixel_t> color_buffer;
	auto color_bytes = 3 * width * height * sizeof(img::pixel_t);
	device_malloc(color_buffer, color_bytes);

	img::device_image_t d_src_img;
	img::make_image(d_src_img, width, height, color_buffer);

	img::device_image_t d_src2_img;
	img::make_image(d_src2_img, width, height, color_buffer);

	img::device_image_t d_dst_img;
	img::make_image(d_dst_img, width, height, color_buffer);


	// setup device memory for gray images
	DeviceBuffer<img::gray::pixel_t> gray_buffer;
	auto gray_bytes = 3 * width * height * sizeof(img::gray::pixel_t);
	device_malloc(gray_buffer, gray_bytes);

	img::gray::device_image_t d_src_gray_img;
	img::make_image(d_src_gray_img, width, height, gray_buffer);

	img::gray::device_image_t d_dst_gray_img;
	img::make_image(d_dst_gray_img, width, height, gray_buffer);

	img::gray::device_image_t d_tmp_gray_img;
	img::make_image(d_tmp_gray_img, width, height, gray_buffer);


	// alpha blend
	img::seq::copy(caddy_img, src_img);
	img::seq::transform_alpha(src_img, [](auto& p){ return 128; });

	img::copy_to_device(src_img, d_src_img);
	img::copy_to_device(corvette_img, d_src2_img);
	img::alpha_blend(d_src_img, d_src2_img, d_dst_img);
	img::copy_to_host(d_dst_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend.png");

	img::copy_to_device(corvette_img, d_dst_img);
	img::alpha_blend(d_src_img, d_dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst.png");


	// grayscale
	img::copy_to_device(caddy_img, d_src_img);
	img::transform_grayscale(d_src_img, d_dst_gray_img);
	img::copy_to_host(d_dst_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "grayscale.png");


	// set converted grayscale as device src image
	img::copy_to_device(dst_gray_img, d_src_gray_img);


	// binarize	
	img::binarize(d_src_gray_img, d_dst_gray_img, 100);
	img::copy_to_host(d_dst_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "binarize.png");


	// convolution kernels
	DeviceBuffer<r32> kernel_buffer;
	auto kernel_bytes = 70 * sizeof(r32);
	device_malloc(kernel_buffer, kernel_bytes);

	img::BlurKernels blur_k;
	img::make_blur_kernels(blur_k, kernel_buffer);

	img::GradientKernels grad_k;
	img::make_gradient_kernels(grad_k, kernel_buffer);


	// blur
	img::blur(d_src_gray_img, d_dst_gray_img, blur_k);
	img::copy_to_host(d_dst_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur.png");


	// edge detection
	u8 threshold = 100;
	img::edges(d_src_gray_img, d_dst_gray_img, threshold, d_tmp_gray_img, blur_k, grad_k);
	img::copy_to_host(d_dst_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "edges.png");


	// gradients
	img::gradients(d_src_gray_img, d_dst_gray_img, d_tmp_gray_img, blur_k, grad_k);
	img::copy_to_host(d_dst_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "gradients.png");


	// recycle memory
	DeviceBuffer<img::gray::pixel_t> sub_buffer;
	sub_buffer.data = d_tmp_gray_img.data;
	sub_buffer.total_bytes = width * height * sizeof(img::gray::pixel_t);
	img::gray::device_image_t d_src_sub;
	img::gray::device_image_t d_dst_sub;
	img::gray::device_image_t d_tmp_sub;
	img::make_image(d_src_sub, width / 2, height / 2, sub_buffer);
	img::make_image(d_dst_sub, width / 2, height / 2, sub_buffer);
	img::make_image(d_tmp_sub, width / 2, height / 2, sub_buffer);


	img::copy_to_host(d_src_gray_img, src_gray_img);
	img::for_each_pixel(dst_gray_img, [](auto& p){ p = 255; });
	
	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_img, range);
	auto dst_sub = img::sub_view(dst_gray_img, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::binarize(d_src_sub, d_dst_sub, 100);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::blur(d_src_sub, d_dst_sub, blur_k);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::gradients(d_src_sub, d_dst_sub, d_tmp_sub, blur_k, grad_k);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::edges(d_src_sub, d_dst_sub, threshold, d_tmp_sub, blur_k, grad_k);	
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 4;
	range.x_end = range.x_begin + width / 2;
	range.y_begin = height / 4;
	range.y_end = range.y_begin + height / 2;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::transform_contrast(d_src_sub, d_dst_sub, 20, 100);
	img::copy_to_host(d_dst_sub, dst_sub);

	img::write_image(dst_gray_img, out_dir + "combo.png");


	device_free(color_buffer);
	device_free(gray_buffer);
	device_free(kernel_buffer);
}


void empty_dir(path_t& dir)
{
	auto last = dir[dir.length() - 1];
	if(last != '/')
	{
		dir += '/';
	}

	std::string command = std::string("mkdir -p ") + dir;
	system(command.c_str());

	command = std::string("rm -rfv ") + dir + '*';
	system(command.c_str());	
}


void print(img::stats_t const& stats)
{
	std::cout << "mean = " << (double)stats.mean << " sigma = " << (double)stats.std_dev << '\n';
}