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
void device_buffer_tests(path_t& dir);
void print(img::stats_t const& stats);

int main()
{
	auto dst_root = DST_IMAGE_ROOT;

	//auto dst_proc = dst_root + "proc/";
    //process_tests(dst_proc);

	auto dst_cuda = dst_root + "cuda/";
	cuda_tests(dst_cuda);

	//auto dst_buffer = dst_root + "cuda/";
	//device_buffer_tests(dst_root);

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

	Image caddy_img;
	img::read_image_from_file(CADILLAC_PATH, caddy_img);
	auto width = caddy_img.width;
	auto height = caddy_img.height;

	Image img_read;
	img::read_image_from_file(CORVETTE_PATH, img_read);
	Image corvette_img;
	corvette_img.width = width;
	corvette_img.height = height;
	img::resize_image(img_read, corvette_img);

	Image src_img;
	img::make_image(src_img, width, height);

	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage src_gray_img;
	img::make_image(src_gray_img, width, height);

	GrayImage dst_gray_img;
	img::make_image(dst_gray_img, width, height);


	// pre-allocate device memory
	u32 pixels_per_image = width * height;
	u32 max_color_images = 4;
	u32 color_bytes = max_color_images * pixels_per_image * sizeof(img::pixel_t);

	DeviceBuffer d_buffer;
	device_malloc(d_buffer, color_bytes);


	


	// grayscale
	img::cuda::transform_grayscale(caddy_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "grayscale.png");	

	img::cuda::transform_grayscale(caddy_img, dst_gray_img, d_buffer);
	img::write_image(dst_gray_img, out_dir + "grayscale_buffer.png");

	img::seq::copy(dst_gray_img, src_gray_img);

	// edge detection
	img::seq::transform_self(dst_gray_img, [](auto p){ return 255;});
	u8 threshold = 100;
	img::cuda::edges(src_gray_img, dst_gray_img, threshold, d_buffer);
	img::write_image(dst_gray_img, out_dir + "edges_buffer.png");


	// binarize
	img::cuda::binarize(src_gray_img, dst_gray_img, 100);
	img::write_image(dst_gray_img, out_dir + "binarize.png");

	img::cuda::binarize(src_gray_img, dst_gray_img, 100, d_buffer);
	img::write_image(dst_gray_img, out_dir + "binarize_buffer.png");


	// alpha blend
	img::seq::copy(caddy_img, src_img);
	img::seq::transform_alpha(src_img, [](auto& p){ return 128; });

	img::cuda::alpha_blend(src_img, corvette_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend.png");

	img::cuda::alpha_blend(src_img, corvette_img, dst_img, d_buffer);
	img::write_image(dst_img, out_dir + "alpha_blend_buffer.png");

	img::seq::copy(corvette_img, dst_img);
	img::cuda::alpha_blend(src_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst.png");

	img::seq::copy(corvette_img, dst_img);
	img::cuda::alpha_blend(src_img, dst_img, d_buffer);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst_buffer.png");


	// blur
	img::cuda::blur(src_gray_img, dst_gray_img, d_buffer);
	img::write_image(dst_gray_img, out_dir + "blur_buffer.png");

	img::cuda::blur(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur.png");


	


	// gradients
	img::seq::transform_self(dst_gray_img, [](auto p){ return 75;});
	img::cuda::gradients(src_gray_img, dst_gray_img, d_buffer);
	img::write_image(dst_gray_img, out_dir + "gradients_buffer.png");


	device_free(d_buffer);

	/*
	// compare edge detection speeds
	// TODO: with cuda

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::data_color_t seq_times;
	seq_times.color = green;

	img::data_color_t par_times;
	par_times.color = blue;

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size = size_start;
	auto const scale = [&](auto t) { return static_cast<r32>(10000 * t / size); };

	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		GrayImage image;
		make_image(image, size);
		GrayImage dst;
		make_image(dst, size);
		auto view = img::make_view(image);
		auto dst_view = img::make_view(dst);

		sw.start();
		img::seq::edges(view, dst_view, 150);
		auto t = sw.get_time_milli();
		seq_times.data.push_back(scale(t));

		sw.start();
		img::edges(view, dst_view, 150);
		t = sw.get_time_milli();
		par_times.data.push_back(scale(t));
	}

	Image view_chart;
	std::vector<img::data_color_t> view_data =
	{
		seq_times, par_times
	};

	img::draw_bar_chart(view_data, view_chart);
	img::write_image(view_chart, out_dir / "edges_times.png");

	*/
}


void device_buffer_tests(path_t& out_dir)
{
	std::cout << "device buffer:\n";
	empty_dir(out_dir);

	DeviceBuffer buffer;
	device_malloc(buffer, 10'000);

	std::array<u32, 10> u32array { 1, 2, 100, 69, 98, 33, 55, 44, 88, 63 };
	DeviceArray<u32> d_u32array;
	push_array(d_u32array, buffer, 10);
	memcpy_to_device(u32array.data(), d_u32array);
	pop_array(d_u32array, buffer);

	std::array<r32, 10> r32array { 1.0f, 2.5f, 100.2f, 69.3f, 98.7f, 33.3f, 55.5f, 44.2f, 88.2f, 63.1f };
	DeviceArray<r32> d_r32array;
	push_array(d_r32array, buffer, 10);
	memcpy_to_device(r32array.data(), d_r32array);
	pop_array(d_r32array, buffer);

	std::array<u8, 10> u8array { 1, 2, 100, 69, 98, 33, 55, 44, 88, 63 };
	DeviceArray<u8> d_u8array;
	push_array(d_u8array, buffer, 10);
	memcpy_to_device(u8array.data(), d_u8array);
	pop_array(d_u8array, buffer);

	

	


	device_free(buffer);
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