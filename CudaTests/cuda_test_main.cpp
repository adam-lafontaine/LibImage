#include "../libimage/libimage.hpp"
#include "../libimage/proc/process.hpp"
#include "../libimage/math/math.hpp"
#include "../libimage/math/charts.hpp"
#include "../libimage/cuda/process.hpp"
#include "./utils/stopwatch.hpp"

#include <cstdio>
#include <string>

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

// make sure these files exist
constexpr auto ROOT_DIR = "/home/adam/Repos/LibImage/CudaTests/";

const auto ROOT_PATH = std::string(ROOT_DIR);

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


void alpha_blend_test(Image const& src, Image const& cur, Image const& dst, path_t const& out_dir)
{
	img::seq::transform_alpha(src, [](auto const& p) { return 128; });

	img::seq::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir + "alpha_blend.png");

	img::simd::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir + "simd_alpha_blend.png");

	img::seq::copy(cur, dst);
	img::seq::alpha_blend(src, dst);
	img::write_image(dst, out_dir + "alpha_blend_src_dst.png");

	img::seq::copy(cur, dst);
	img::simd::alpha_blend(src, dst);
	img::write_image(dst, out_dir + "simd_alpha_blend_src_dst.png");

	img::seq::transform_alpha(src, [](auto const& p) { return 255; });
}


void grayscale_test(Image const& src, GrayImage const& dst, path_t const& out_dir)
{
	img::seq::grayscale(src, dst);	
	img::write_image(dst, out_dir + "grayscale.png");

	img::simd::grayscale(src, dst);
	img::write_image(dst, out_dir + "simd_grayscale.png");
}


void rotate_test(Image const& src, Image const& dst, path_t const& out_dir)
{
	r32 theta = 0.6f * 2 * 3.14159f;
	img::seq::rotate(src, dst, src.width / 2, src.height / 2, theta);
	img::write_image(dst, out_dir + "rotate.png");
}


void rotate_gray_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	r32 theta = 0.6f * 2 * 3.14159f;
	img::seq::rotate(src, dst, src.width / 2, src.height / 2, theta);
	img::write_image(dst, out_dir + "rotate_gray.png");
}


void stats_test(GrayImage const& src, path_t const& out_dir)
{
	auto gray_stats = img::calc_stats(src);
	GrayImage gray_stats_img;
	img::draw_histogram(gray_stats.hist, gray_stats_img);
	img::write_image(gray_stats_img, out_dir + "gray_stats.png");
	print(gray_stats);

	gray_stats_img.dispose();
}


void alpha_grayscale_test(Image const& src, path_t const& out_dir)
{
	img::seq::alpha_grayscale(src);
	auto alpha_stats = img::calc_stats(src, img::Channel::Alpha);
	GrayImage alpha_stats_img;
	img::draw_histogram(alpha_stats.hist, alpha_stats_img);
	img::write_image(alpha_stats_img, out_dir + "alpha_stats.png");
	print(alpha_stats);

	alpha_stats_img.dispose();
	img::seq::transform_alpha(src, [](auto const& p) { return 255; });
}


void contrast_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	auto gray_stats = img::calc_stats(src);
	auto shade_min = (u8)(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = (u8)(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));

	img::seq::contrast(src, dst, shade_min, shade_max);
	img::write_image(dst, out_dir + "contrast.png");
}


void binarize_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	auto gray_stats = img::calc_stats(src);
	auto const is_white = [&](u8 p) { return (r32)(p) > gray_stats.mean; };

	img::seq::binarize(src, dst, is_white);
	img::write_image(dst, out_dir + "binarize.png");
}


void blur_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	img::seq::blur(src, dst);
	img::write_image(dst, out_dir + "blur.png");

	img::simd::blur(src, dst);
	img::write_image(dst, out_dir + "simd_blur.png");
}


void edges_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	auto const threshold = [](u8 g) { return g >= 100; };

	img::seq::edges(src, dst, threshold);
	img::write_image(dst, out_dir + "edges.png");

	img::simd::edges(src, dst, threshold);
	img::write_image(dst, out_dir + "simd_edges.png");
}


void gradients_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	img::seq::gradients(src, dst);
	img::write_image(dst, out_dir + "gradient.png");

	img::simd::gradients(src, dst);
	img::write_image(dst, out_dir + "simd_gradient.png");
}


void combine_views_test(GrayImage const& src, GrayImage const& dst, path_t const& out_dir)
{
	auto width = src.width;
	auto height = src.height;

	auto gray_stats = img::calc_stats(src);
	auto shade_min = (u8)(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = (u8)(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));

	auto const is_white = [&](u8 p) { return (r32)(p) > gray_stats.mean; };
	auto const threshold = [](u8 g) { return g >= 100; };

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src, range);
	auto dst_sub = img::sub_view(dst, range);
	img::seq::gradients(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::seq::contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::seq::blur(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::seq::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 4;
	range.x_end = range.x_begin + width / 2;
	range.y_begin = height / 4;
	range.y_end = range.y_begin + height / 2;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::seq::edges(src_sub, dst_sub, threshold);

	img::write_image(dst, out_dir + "combo.png");
}


void process_tests(path_t& out_dir)
{
	// C++17 not available on Jetson Nano.
	// No stl parallel algorithms

	printf("\nprocess:\n");
	empty_dir(out_dir);

	// get image
	Image corvette;
	img::read_image_from_file(CORVETTE_PATH, corvette);

	auto const width = corvette.width;
	auto const height = corvette.height;	

	// get another image for blending
	// make sure it is the same size
	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	Image caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(caddy_read, caddy);
	img::write_image(caddy, out_dir + "caddy.png");

	// read a grayscale image
	GrayImage src_gray;
	img::read_image_from_file(CORVETTE_PATH, src_gray);


	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage dst_gray;
	img::make_image(dst_gray, width, height);


	alpha_blend_test(caddy, corvette, dst_img, out_dir);	

	grayscale_test(corvette, dst_gray, out_dir);

	rotate_test(caddy, dst_img, out_dir);

	rotate_gray_test(src_gray, dst_gray, out_dir);

	stats_test(src_gray, out_dir);

	alpha_grayscale_test(corvette, out_dir);

	contrast_test(src_gray, dst_gray, out_dir);

	binarize_test(src_gray, dst_gray, out_dir);

	blur_test(src_gray, dst_gray, out_dir);

	edges_test(src_gray, dst_gray, out_dir);

	gradients_test(src_gray, dst_gray, out_dir);
	
	combine_views_test(src_gray, dst_gray, out_dir);
}


void cuda_alpha_blend_test(
	Image const& src, Image const& cur, Image const& dst, 
	CudaImage const& d_src, CudaImage const& d_cur, CudaImage const& d_dst, 
	path_t const& out_dir
	)
{	
	img::seq::transform_alpha(src, [](auto& p){ return 128; });

	img::copy_to_device(src, d_src);
	img::copy_to_device(cur, d_cur);
	img::alpha_blend(d_src, d_cur, d_dst);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "alpha_blend.png");

	img::copy_to_device(cur, d_dst);
	img::alpha_blend(d_src, d_dst);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "alpha_blend_src_dst.png");

	img::seq::transform_alpha(src, [](auto& p){ return 255; });
}


void cuda_grayscale_test(
	Image const& src, GrayImage const& dst, 
	CudaImage const& d_src, CudaGrayImage const& d_dst, 
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);
	img::grayscale(d_src, d_dst);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "grayscale.png");
}


void cuda_rotate_test(
	Image const& src, Image const& dst, 
	CudaImage const& d_src, CudaImage const& d_dst, 
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);

	auto origin_x = d_src.width / 2;
	auto origin_y = d_src.height / 2;
	r32 theta = 0.6f * 2 * 3.14159f;

	img::rotate(d_src, d_dst, origin_x, origin_y, theta);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "rotate.png");
}


void cuda_rotate_gray_test(
	GrayImage const& src, GrayImage const& dst, 
	CudaGrayImage const& d_src, CudaGrayImage const& d_dst, 
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);

	auto origin_x = d_src.width / 2;
	auto origin_y = d_src.height / 2;
	r32 theta = 0.6f * 2 * 3.14159f;

	img::rotate(d_src, d_dst, origin_x, origin_y, theta);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "rotate_gray.png");
}


void cuda_binarize_test(
	GrayImage const& src, GrayImage const& dst, 
	CudaGrayImage const& d_src, CudaGrayImage const& d_dst, 
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);
	img::binarize(d_src, d_dst, 100);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "binarize.png");
}


void cuda_blur_test(
	GrayImage const& src, GrayImage const& dst, 
	CudaGrayImage const& d_src, CudaGrayImage const& d_dst, 
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);
	img::blur(d_src, d_dst);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "blur.png");
}


void cuda_edges_test(
	GrayImage const& src, GrayImage const& dst, 
	CudaGrayImage const& d_src, CudaGrayImage const& d_dst, CudaGrayImage d_temp,
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);
	u8 threshold = 100;
	img::edges(d_src, d_dst, threshold, d_temp);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "edges.png");
}


void cuda_gradients_test(
	GrayImage const& src, GrayImage const& dst, 
	CudaGrayImage const& d_src, CudaGrayImage const& d_dst, CudaGrayImage d_temp,
	path_t const& out_dir
	)
{
	img::copy_to_device(src, d_src);
	img::gradients(d_src, d_dst, d_temp);
	img::copy_to_host(d_dst, dst);
	img::write_image(dst, out_dir + "edges.png");
}


void cuda_combine_views_test(
	GrayImage const& src, GrayImage const& dst,
	DeviceBuffer sub_buffer,
	path_t const& out_dir
	)
{
	auto width = src.width;
	auto height = src.height;
	u8 threshold = 100;

	CudaGrayImage d_src_sub;
	CudaGrayImage d_dst_sub;
	CudaGrayImage d_tmp_sub;
	img::make_image(d_src_sub, width / 2, height / 2, sub_buffer);
	img::make_image(d_dst_sub, width / 2, height / 2, sub_buffer);
	img::make_image(d_tmp_sub, width / 2, height / 2, sub_buffer);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src, range);
	auto dst_sub = img::sub_view(dst, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::binarize(d_src_sub, d_dst_sub, 100);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::blur(d_src_sub, d_dst_sub);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::gradients(d_src_sub, d_dst_sub, d_tmp_sub);
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::edges(d_src_sub, d_dst_sub, threshold, d_tmp_sub);	
	img::copy_to_host(d_dst_sub, dst_sub);

	range.x_begin = width / 4;
	range.x_end = range.x_begin + width / 2;
	range.y_begin = height / 4;
	range.y_end = range.y_begin + height / 2;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);
	img::copy_to_device(src_sub, d_src_sub);
	img::contrast(d_src_sub, d_dst_sub, 20, 100);
	img::copy_to_host(d_dst_sub, dst_sub);

	img::write_image(dst, out_dir + "combo.png");
}


void cuda_tests(path_t& out_dir)
{
	printf("\ncuda:\n");
	empty_dir(out_dir);

	Image corvette;
	img::read_image_from_file(CORVETTE_PATH, corvette);
	auto width = corvette.width;
	auto height = corvette.height;

	Image img_read;
	img::read_image_from_file(CADILLAC_PATH, img_read);

	Image caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(img_read, caddy);

	// read a grayscale image
	GrayImage src_gray;
	img::read_image_from_file(CORVETTE_PATH, src_gray);

	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage dst_gray_img;
	img::make_image(dst_gray_img, width, height);
	
	
	// setup device memory for color images
	DeviceBuffer color_buffer;
	auto color_bytes = 3 * width * height * PIXEL_SZ;
	device_malloc(color_buffer, color_bytes);

	CudaImage d_src_img;
	img::make_image(d_src_img, width, height, color_buffer);

	CudaImage d_src2_img;
	img::make_image(d_src2_img, width, height, color_buffer);

	CudaImage d_dst_img;
	img::make_image(d_dst_img, width, height, color_buffer);


	// setup device memory for gray images
	DeviceBuffer gray_buffer;
	auto gray_bytes = 3 * width * height * G_PIXEL_SZ;
	device_malloc(gray_buffer, gray_bytes);

	CudaGrayImage d_src_gray_img;
	img::make_image(d_src_gray_img, width, height, gray_buffer);

	CudaGrayImage d_dst_gray_img;
	img::make_image(d_dst_gray_img, width, height, gray_buffer);

	CudaGrayImage d_tmp_gray_img;
	img::make_image(d_tmp_gray_img, width, height, gray_buffer);


	cuda_alpha_blend_test(caddy, corvette, dst_img, d_src_img, d_src2_img, d_dst_img, out_dir);

	cuda_grayscale_test(corvette, dst_gray_img, d_src_img, d_dst_gray_img, out_dir);

	cuda_rotate_test(caddy, dst_img, d_src_img, d_dst_img, out_dir);

	cuda_rotate_gray_test(src_gray, dst_gray_img, d_src_gray_img, d_dst_gray_img, out_dir);

	cuda_binarize_test(src_gray, dst_gray_img, d_src_gray_img, d_dst_gray_img, out_dir);

	cuda_blur_test(src_gray, dst_gray_img, d_src_gray_img, d_dst_gray_img, out_dir);	
	
	cuda_edges_test(src_gray, dst_gray_img, d_src_gray_img, d_dst_gray_img, d_tmp_gray_img, out_dir);

	cuda_gradients_test(src_gray, dst_gray_img, d_src_gray_img, d_dst_gray_img, d_tmp_gray_img, out_dir);	


	// recycle memory
	DeviceBuffer sub_buffer;
	sub_buffer.data = d_tmp_gray_img.data;
	sub_buffer.total_bytes = width * height * G_PIXEL_SZ;

	cuda_combine_views_test(src_gray, dst_gray_img, sub_buffer, out_dir);	

	device_free(color_buffer);
	device_free(gray_buffer);
}


void gradient_times(path_t& out_dir)
{
	printf("\ngradients:\n");
	empty_dir(out_dir);

	u32 n_image_sizes = 2;
	u32 image_dim_factor = 4;

	u32 n_image_counts = 2;
	u32 image_count_factor = 4;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 100;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);
	auto red = img::to_pixel(192, 40, 40);

	img::multi_chart_data_t seq_times;
	seq_times.color = green;

	img::multi_chart_data_t simd_times;
	simd_times.color = blue;

	img::multi_chart_data_t gpu_times;
	gpu_times.color = red;

	Stopwatch sw;
	u32 width = width_start;
	u32 height = height_start;
	u32 image_count = image_count_start;

	auto const current_pixels = [&]() { return (u64)(width) * height * image_count; };

	auto const start_pixels = current_pixels();

	auto const scale = [&](auto t) { return (r32)(start_pixels) / current_pixels() * t; };
	auto const print_wh = [&]() { printf("\nwidth: %u height: %u\n", width, height); };
	auto const print_count = [&]() { printf("  image count: %u\n", image_count); };

	r64 t = 0;
	auto const print_t = [&](const char* label) { printf("    %s time: %f\n", label, scale(t)); };

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;
		std::vector<r32> seq;
		std::vector<r32> simd;
		std::vector<r32> gpu;
		GrayImage src;
		GrayImage dst;
		GrayImage tmp;
		img::make_image(src, width, height);
		img::make_image(dst, width, height);
		img::make_image(tmp, width, height);

		DeviceBuffer d_buffer;
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
			print_t(" seq");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::simd::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			simd.push_back(scale(t));
			print_t("simd");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::copy_to_device(src, d_src);
				img::gradients(d_src, d_dst, d_tmp);
				img::copy_to_host(d_dst, dst);
			}
			t = sw.get_time_milli();
			gpu.push_back(scale(t));
			print_t(" gpu");

			image_count *= image_count_factor;
		}

		device_free(d_buffer);

		seq_times.data_list.push_back(seq);
		simd_times.data_list.push_back(simd);
		gpu_times.data_list.push_back(gpu);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}

	img::grouped_multi_chart_data_t chart_data
	{ 
		seq_times, gpu_times
	};
	Image chart;
	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir + "gradients.bmp");
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
	printf("mean = %f sigma = %f\n", stats.mean, stats.std_dev);
}