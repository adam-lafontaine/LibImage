#include "../libimage/libimage.hpp"
#include "../libimage/proc/process.hpp"
#include "../libimage/math/math.hpp"
#include "../libimage/math/charts.hpp"
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
using Pixel = img::pixel_t;

using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using GrayPixel = img::gray::pixel_t;

using CudaImage = img::device_image_t;
using CudaGrayImage = img::gray::device_image_t;

constexpr auto PIXEL_SZ = sizeof(Pixel);
constexpr auto G_PIXEL_SZ = sizeof(GrayPixel);

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
void gradient_times(path_t& out_dir);
void print(img::stats_t const& stats);

int main()
{
	auto dst_root = DST_IMAGE_ROOT;

	auto dst_proc = dst_root + "proc/";
    process_tests(dst_proc);

	auto dst_cuda = dst_root + "cuda/";
	cuda_tests(dst_cuda);

	auto dst_timing = dst_root + "timing/";
	gradient_times(dst_timing);

    printf("\nDone.\n");
}


void process_tests(path_t& out_dir)
{
	// C++17 not available on Jetson Nano.
	// No stl parallel algorithms

	std::cout << "\nprocess:\n";
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
	u32 threshold = 150;
	img::seq::edges(src_gray_img, dst_gray_img, threshold);
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
	img::seq::gradients(src_sub, dst_sub);
	

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
	img::seq::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 4;
	range.x_end = range.x_begin + width / 2;
	range.y_begin = height / 4;
	range.y_end = range.y_begin + height / 2;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::seq::edges(src_sub, dst_sub, threshold);

	img::write_image(dst_gray_img, out_dir + "combo.png");


}


void cuda_tests(path_t& out_dir)
{
	std::cout << "\ncuda:\n";
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
	DeviceBuffer<Pixel> color_buffer;
	auto color_bytes = 3 * width * height * PIXEL_SZ;
	device_malloc(color_buffer, color_bytes);

	CudaImage d_src_img;
	img::make_image(d_src_img, width, height, color_buffer);

	CudaImage d_src2_img;
	img::make_image(d_src2_img, width, height, color_buffer);

	CudaImage d_dst_img;
	img::make_image(d_dst_img, width, height, color_buffer);


	// setup device memory for gray images
	DeviceBuffer<GrayPixel> gray_buffer;
	auto gray_bytes = 3 * width * height * G_PIXEL_SZ;
	device_malloc(gray_buffer, gray_bytes);

	CudaGrayImage d_src_gray_img;
	img::make_image(d_src_gray_img, width, height, gray_buffer);

	CudaGrayImage d_dst_gray_img;
	img::make_image(d_dst_gray_img, width, height, gray_buffer);

	CudaGrayImage d_tmp_gray_img;
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
	DeviceBuffer<GrayPixel> sub_buffer;
	sub_buffer.data = d_tmp_gray_img.data;
	sub_buffer.total_bytes = width * height * G_PIXEL_SZ;
	CudaGrayImage d_src_sub;
	CudaGrayImage d_dst_sub;
	CudaGrayImage d_tmp_sub;
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


void gradient_times(path_t& out_dir)
{
	std::cout << "\ngradients:\n";
	empty_dir(out_dir);

	u32 n_image_sizes = 3;
	u32 image_dim_factor = 2;

	u32 n_image_counts = 5;
	u32 image_count_factor = 2;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 10;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::multi_chart_data_t seq_times;
	seq_times.color = green;

	img::multi_chart_data_t gpu_times;
	gpu_times.color = green;

	Stopwatch sw;
	u32 width = width_start;
	u32 height = height_start;
	u32 image_count = image_count_start;

	auto const current_pixels = [&]() { return static_cast<r64>(width) * height * image_count; };

	auto const start_pixels = current_pixels();

	auto const scale = [&](auto t) { return static_cast<r32>(start_pixels / current_pixels() * t); };
	auto const print_wh = [&]() { std::cout << "\nwidth: " << width << " height: " << height << '\n'; };
	auto const print_count = [&]() { std::cout << "  image count: " << image_count << '\n'; };

	r64 t = 0;
	auto const print_t = [&](const char* label) { std::cout << "    " << label << " time: " << scale(t) << '\n'; };

	DeviceBuffer<r32> kernel_buffer;
	auto kernel_bytes = 70 * sizeof(r32);
	device_malloc(kernel_buffer, kernel_bytes);

	img::BlurKernels blur_k;
	img::make_blur_kernels(blur_k, kernel_buffer);

	img::GradientKernels grad_k;
	img::make_gradient_kernels(grad_k, kernel_buffer);

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;
		std::vector<r32> seq;
		std::vector<r32> gpu;
		GrayImage src;
		GrayImage dst;
		GrayImage tmp;
		img::make_image(src, width, height);
		img::make_image(dst, width, height);
		img::make_image(tmp, width, height);

		DeviceBuffer<GrayPixel> d_buffer;
		auto gray_bytes = 3 * width * height * G_PIXEL_SZ;
		device_malloc(d_buffer, gray_bytes);

		CudaGrayImage d_src;
		CudaGrayImage d_dst;
		CudaGrayImage d_tmp;
		img::make_image(d_src, width, height, d_buffer);
		img::make_image(d_dst, width, height, d_buffer);
		img::make_image(d_tmp, width, height, d_buffer);

		for (u32 c = 0; c < n_image_counts; ++c)
		{
			print_count();

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::seq::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			seq.push_back(scale(t));
			print_t("seq");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::copy_to_device(src, d_src);
				img::gradients(d_src, d_dst, d_tmp, blur_k, grad_k);
				img::copy_to_host(d_dst, dst);
			}
			t = sw.get_time_milli();
			gpu.push_back(scale(t));
			print_t("gpu");

			image_count *= image_count_factor;

		}

		device_free(d_buffer);

		seq_times.data_list.push_back(seq);
		gpu_times.data_list.push_back(gpu);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}

	device_free(kernel_buffer);

	img::grouped_multi_chart_data_t chart_data
	{ 
		seq_times, gpu_times
	};
	Image chart;
	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir + "gradients.bmp");
}


void do_gradients(GrayImage const& src, GrayImage& dst, u32 qty)
{
	auto const width = src.width;
	auto const height = src.height;

	GrayImage tmp;
	img::make_image(tmp, width, height);

	for(u32 i = 0; i < qty; ++i)
	{
		img::seq::gradients(src, dst, tmp);
	}
}


void cuda_do_gradients(GrayImage const& src, GrayImage& dst, u32 qty)
{
	auto const width = src.width;
	auto const height = src.height;

	DeviceBuffer<GrayPixel> d_buffer;
	auto gray_bytes = 3 * src.width * src.height * G_PIXEL_SZ;
	device_malloc(d_buffer, gray_bytes);

	CudaGrayImage d_src;
	CudaGrayImage d_dst;
	CudaGrayImage d_tmp;

	img::make_image(d_src, width, height, d_buffer);
	img::make_image(d_dst, width, height, d_buffer);
	img::make_image(d_tmp, width, height, d_buffer);

	// convolution kernels
	DeviceBuffer<r32> kernel_buffer;
	auto kernel_bytes = 70 * sizeof(r32);
	device_malloc(kernel_buffer, kernel_bytes);

	img::BlurKernels blur_k;
	img::make_blur_kernels(blur_k, kernel_buffer);

	img::GradientKernels grad_k;
	img::make_gradient_kernels(grad_k, kernel_buffer);

	for(u32 i = 0; i < qty; ++i)
	{
		img::copy_to_device(src, d_src);

		img::gradients(d_src, d_dst, d_tmp, blur_k, grad_k);

		img::copy_to_host(d_dst, dst);
	}

	device_free(d_buffer);
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