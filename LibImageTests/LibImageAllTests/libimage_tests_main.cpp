#include "../../libimage/libimage.hpp"
#include "../../libimage/math/math.hpp"
#include "../../libimage/math/charts.hpp"
#include "../../libimage/proc/process.hpp"

#include "../utils/stopwatch.hpp"

//#define CHECK_LEAKS

#if defined(_WIN32) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

#include <iostream>
#include <random>
#include <algorithm>
#include <execution>
#include <functional>
#include <filesystem>

namespace fs = std::filesystem;
namespace img = libimage;

using Image = img::image_t;
using ImageView = img::view_t;
using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using Pixel = img::pixel_t;


// set this directory for your system
constexpr auto ROOT_DIR = "C:/D_Data/repos/LibImage/LibImageTests/LibImageAllTests";

const auto ROOT_PATH = fs::path(ROOT_DIR);

// make sure these files exist
const auto CORVETTE_PATH = ROOT_PATH / "in_files/png/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH / "in_files/png/cadillac.png";
const auto RED_PATH = ROOT_PATH / "in_files/bmp/red.bmp";

const auto SRC_IMAGE_PATH = CORVETTE_PATH;
const auto DST_IMAGE_ROOT = ROOT_PATH / "out_files";

void empty_dir(fs::path const& dir);
void print(ImageView const& view);
void print(GrayView const& view);
void print(img::stats_t const& stats);

void basic_tests(fs::path const& out_dir);
void math_tests(fs::path const& out_dir);
void for_each_tests(fs::path const& out_dir);
void transform_tests(fs::path const& out_dir);
void process_tests(fs::path const& out_dir);
void gradient_times(fs::path const& out_dir);


int main()
{
#if defined(_WIN32) && defined(_DEBUG) && defined(CHECK_LEAKS)
	int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
	dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
	dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
	_CrtSetDbgFlag(dbgFlags);
#endif

	auto dst_root = fs::path(DST_IMAGE_ROOT);	

	/*basic_tests(dst_root / "basic");
	math_tests(dst_root / "math");*/

	process_tests(dst_root / "process");

	auto timing_dir = dst_root / "timing";
	empty_dir(timing_dir);

	/*for_each_tests(timing_dir);
	transform_tests(timing_dir);*/

	//gradient_times(timing_dir);

	std::cout << "\nDone.\n";
}


void basic_tests(fs::path const& out_dir)
{
	std::cout << "basic:\n";
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);

	// write different file types
	img::write_image(image, out_dir / "image.png");
	img::write_image(image, out_dir / "image.bmp");

	// write a view from an image file
	Image red_image;
	img::read_image_from_file(RED_PATH, red_image);
	auto red_view = img::make_view(red_image);
	img::write_image(red_image, out_dir / "red_image.png");
	img::write_view(red_view, out_dir / "red_view.png");

	auto view = img::make_view(image);
	print(view);

	// write views with different file types
	img::write_view(view, out_dir / "view.png");
	img::write_view(view, out_dir / "view.bmp");

	auto w = view.width;
	auto h = view.height;

	// get a portion of an existing image
	img::pixel_range_t range = { w * 1 / 3, w * 2 / 3, h * 1 / 3, h * 2 / 3 };
	auto sub_view = img::sub_view(view, range);
	print(sub_view);
	img::write_view(sub_view, out_dir / "sub.png");

	// get one row from an image
	auto row_view = img::row_view(view, h / 2);
	print(row_view);
	img::write_view(row_view, out_dir / "row_view.bmp");

	// get one column from an image
	auto col_view = img::column_view(view, w / 2);
	print(col_view);
	img::write_view(col_view, out_dir / "col_view.bmp");

	// resize an image
	Image resize_image;
	resize_image.width = w / 4;
	resize_image.height = h / 2;
	auto resize_view = img::make_resized_view(image, resize_image);
	print(resize_view);
	img::write_image(resize_image, out_dir / "resize_image.bmp");
	img::write_view(resize_view, out_dir / "resize_view.bmp");

	// read a color image to grayscale
	GrayImage image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	img::write_image(image_gray, out_dir / "image_gray.bmp");

	// create a grayscale view
	auto view_gray = img::make_view(image_gray);
	print(view_gray);
	img::write_view(view_gray, out_dir / "view_gray.bmp");

	// portion of a grayscale image
	auto sub_view_gray = img::sub_view(view_gray, range);
	print(sub_view_gray);
	img::write_view(sub_view_gray, out_dir / "sub_view_gray.png");

	// row from a grayscale image
	auto row_view_gray = img::row_view(view_gray, view_gray.height / 2);
	print(row_view_gray);
	img::write_view(row_view_gray, out_dir / "row_view_gray.png");

	// column from a grayscale image
	auto col_view_gray = img::column_view(view_gray, view_gray.width / 2);
	print(col_view_gray);
	img::write_view(col_view_gray, out_dir / "col_view_gray.png");

	// resize a grayscale image
	GrayImage resize_image_gray;
	resize_image_gray.width = w / 4;
	resize_image_gray.height = h / 2;
	auto resize_view_gray = img::make_resized_view(image_gray, resize_image_gray);
	print(resize_view_gray);
	img::write_image(resize_image_gray, out_dir / "resize_image_gray.bmp");
	img::write_view(resize_view_gray, out_dir / "resize_view_gray.bmp");

	std::cout << '\n';
}


void math_tests(fs::path const& out_dir)
{
	std::cout << "math:\n";
	empty_dir(out_dir);

	GrayImage image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	auto view_gray = img::make_view(image_gray);

	// get shade histogram, mean and standard deviation from a grayscale image
	auto stats_gray = img::calc_stats(view_gray);

	// write the histogram to a new image
	GrayImage stats_image_gray;
	img::draw_histogram(stats_gray.hist, stats_image_gray);
	print(stats_gray);
	img::write_image(stats_image_gray, out_dir / "stats_image_gray.png");

	Image image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);
	auto view = img::make_view(image);

	// histogram, mean and standard deviation of each rgb channel
	auto stats = img::calc_stats(view);

	// draw each histogram to a new image
	Image stats_image;
	img::draw_histogram(stats, stats_image);
	print(stats.red);
	print(stats.green);
	print(stats.blue);
	img::write_image(stats_image, out_dir / "stats_image.png");	

	// create a grayscale image and set each pixel with a predicate
	auto const binarize = [&](u8 p) { return p > stats_gray.mean ? 255 : 0; };
	GrayImage binary;
	img::make_image(binary, image_gray.width, image_gray.height);
	std::transform(image_gray.begin(), image_gray.end(), binary.begin(), binarize);

	img::write_image(binary, out_dir / "binary.png");

	std::cout << '\n';
}


Pixel alpha_blend_linear(Pixel const& src, Pixel const& current)
{
	auto const to_r32 = [](u8 c) { return static_cast<r32>(c) / 255.0f; };

	auto a = to_r32(src.alpha);

	auto const blend = [&](u8 s, u8 c) 
	{
		auto sf = static_cast<r32>(s);
		auto cf = static_cast<r32>(c);
		
		auto blended = a * cf + (1.0f - a) * sf;

		return static_cast<u8>(blended);
	};

	auto red = blend(src.red, current.red);
	auto green = blend(src.green, current.green);
	auto blue = blend(src.blue, current.blue);

	return img::to_pixel(red, green, blue);
}


void for_each_tests(fs::path const& out_dir)
{
	std::cout << "\nfor_each:\n";

	std::random_device rd;
	std::default_random_engine reng(rd());
	std::uniform_int_distribution<int> dist(0, 255);

	auto const random_pixel = [&]() 
	{
		Pixel p;

		for (u32 i = 0; i < 4; ++i)
		{
			p.channels[i] = static_cast<u8>(dist(reng));
		}

		return p;
	};

	auto const random_blended_pixel = [&](Pixel& p) 
	{
		Pixel src = img::to_pixel(0);// random_pixel();

		p = alpha_blend_linear(src, p);
	};

	u32 n_image_sizes = 2;
	u32 image_dim_factor = 4;

	u32 n_image_counts = 5;
	u32 image_count_factor = 2;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 50;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::multi_chart_data_t loop_image_times;
	loop_image_times.color = green;

	img::multi_chart_data_t loop_view_times;
	loop_view_times.color = blue;

	img::multi_chart_data_t stl_image_times;
	stl_image_times.color = green;

	img::multi_chart_data_t stl_view_times;
	stl_view_times.color = blue;

	img::multi_chart_data_t par_image_times;
	par_image_times.color = green;

	img::multi_chart_data_t par_view_times;
	par_view_times.color = blue;


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

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;

		Image image;
		img::make_image(image, width, height);
		auto view = img::make_view(image);

		std::vector<r32> loop_image;
		std::vector<r32> loop_view;
		std::vector<r32> stl_image;
		std::vector<r32> stl_view;
		std::vector<r32> par_image;
		std::vector<r32> par_view;

		for (u32 c = 0; c < n_image_counts; ++c)
		{
			print_count();			

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::for_each_pixel(image, random_blended_pixel);
			}			
			t = sw.get_time_milli();
			loop_image.push_back(scale(t));
			print_t("loop image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::for_each_pixel(view, random_blended_pixel);
			}			
			t = sw.get_time_milli();
			loop_view.push_back(scale(t));
			print_t(" loop view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::for_each(image.begin(), image.end(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			stl_image.push_back(scale(t));
			print_t(" stl image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::for_each(view.begin(), view.end(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			stl_view.push_back(scale(t));
			print_t("  stl view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::for_each(std::execution::par, image.begin(), image.end(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			par_image.push_back(scale(t));
			print_t(" par image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::for_each(std::execution::par, view.begin(), view.end(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			par_view.push_back(scale(t));
			print_t("  par view");

			image_count *= image_count_factor;
		}

		loop_image_times.data_list.push_back(loop_image);
		loop_view_times.data_list.push_back(loop_view);
		stl_image_times.data_list.push_back(stl_image);
		stl_view_times.data_list.push_back(stl_view);
		par_image_times.data_list.push_back(par_image);
		par_view_times.data_list.push_back(par_view);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}	

	Image chart;
	img::grouped_multi_chart_data_t chart_data
	{
		loop_image_times, loop_view_times,
		stl_image_times, stl_view_times,
		par_image_times, par_view_times,
	};

	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir / "for_each.bmp");
}


void transform_tests(fs::path const& out_dir)
{
	std::cout << "\ntransform:\n";

	std::random_device rd;
	std::default_random_engine reng(rd());
	std::uniform_int_distribution<int> dist(0, 255);

	auto const random_pixel = [&]()
	{
		Pixel p;

		for (u32 i = 0; i < 4; ++i)
		{
			p.channels[i] = static_cast<u8>(dist(reng));
		}

		return p;
	};

	auto const random_blended_pixel = [&](Pixel const& p)
	{
		Pixel src = img::to_pixel(0);// random_pixel();

		return alpha_blend_linear(src, p);
	};

	u32 n_image_sizes = 2;
	u32 image_dim_factor = 4;

	u32 n_image_counts = 5;
	u32 image_count_factor = 2;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 50;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::multi_chart_data_t stl_image_times;
	stl_image_times.color = green;

	img::multi_chart_data_t stl_view_times;
	stl_view_times.color = blue;

	img::multi_chart_data_t par_image_times;
	par_image_times.color = green;

	img::multi_chart_data_t par_view_times;
	par_view_times.color = blue;


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

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;

		Image image;
		img::make_image(image, width, height);
		auto view = img::make_view(image);

		Image dst;
		img::make_image(dst, width, height);

		std::vector<r32> stl_image;
		std::vector<r32> stl_view;
		std::vector<r32> par_image;
		std::vector<r32> par_view;

		for (u32 c = 0; c < n_image_counts; ++c)
		{
			print_count();

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::transform(image.begin(), image.end(), dst.begin(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			stl_image.push_back(scale(t));
			print_t("stl image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::transform(view.begin(), view.end(), dst.begin(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			stl_view.push_back(scale(t));
			print_t(" stl view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::transform(std::execution::par, image.begin(), image.end(), dst.begin(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			par_image.push_back(scale(t));
			print_t("par image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				std::transform(std::execution::par, view.begin(), view.end(), dst.begin(), random_blended_pixel);
			}
			t = sw.get_time_milli();
			par_view.push_back(scale(t));
			print_t(" par view");

			image_count *= image_count_factor;
		}

		stl_image_times.data_list.push_back(stl_image);
		stl_view_times.data_list.push_back(stl_view);
		par_image_times.data_list.push_back(par_image);
		par_view_times.data_list.push_back(par_view);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}

	Image chart;
	img::grouped_multi_chart_data_t chart_data
	{
		stl_image_times, stl_view_times,
		par_image_times, par_view_times,
	};

	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir / "transform.bmp");
}


void process_tests(fs::path const& out_dir)
{
	std::cout << "\nprocess:\n";
	empty_dir(out_dir);

	// get image
	Image corvette_image;
	img::read_image_from_file(CORVETTE_PATH, corvette_image);
	auto const width = corvette_image.width;
	auto const height = corvette_image.height;
	auto corvette_view = img::make_view(corvette_image);

	Image dst_image;
	img::make_image(dst_image, width, height);

	GrayImage dst_gray_image;
	img::make_image(dst_gray_image, width, height);

	// get another image for blending
	// make sure it is the same size
	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	Image caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(caddy_read, caddy);
	auto caddy_view = make_view(caddy);

	// alpha blending
	img::transform_alpha(caddy_view, [](auto const& p) { return 128; });
	img::simd::alpha_blend(caddy_view, corvette_view, dst_image);
	img::write_image(dst_image, out_dir / "alpha_blend.png");

	img::copy(corvette_view, dst_image);
	img::simd::alpha_blend(caddy_view, dst_image);
	img::write_image(dst_image, out_dir / "alpha_blend_src_dst.png");

	// grayscale
	img::simd::grayscale(corvette_view, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "grayscale.png");
	
	// stats
	auto gray_stats = img::calc_stats(dst_gray_image);
	GrayImage gray_stats_image;
	img::draw_histogram(gray_stats.hist, gray_stats_image);
	img::write_image(gray_stats_image, out_dir / "gray_stats.png");
	print(gray_stats);

	// alpha grayscale
	img::transform_alpha_grayscale(corvette_view);
	auto alpha_stats = img::calc_stats(corvette_image, img::Channel::Alpha);
	GrayImage alpha_stats_image;
	img::draw_histogram(alpha_stats.hist, alpha_stats_image);
	img::write_image(alpha_stats_image, out_dir / "alpha_stats.png");
	print(alpha_stats);

	// create a new grayscale source
	GrayImage src_gray_image;
	img::make_image(src_gray_image, width, height);
	img::copy(dst_gray_image, src_gray_image);

	// contrast
	auto shade_min = static_cast<u8>(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = static_cast<u8>(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));
	img::contrast(src_gray_image, dst_gray_image, shade_min, shade_max);
	img::write_image(dst_gray_image, out_dir / "contrast.png");

	// binarize
	auto const is_white = [&](u8 p) { return static_cast<r32>(p) > gray_stats.mean; };
	img::binarize(src_gray_image, dst_gray_image, is_white);
	img::write_image(dst_gray_image, out_dir / "binarize.png");

	//blur
	img::simd::blur(src_gray_image, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "blur.png");	

	// edge detection
	img::simd::edges(src_gray_image, dst_gray_image, [](u8 g) { return g >= 100; });
	img::write_image(dst_gray_image, out_dir / "edges.png");

	// gradient
	img::simd::gradients(src_gray_image, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "gradient.png");

	// combine transformations in the same image
	// regular grayscale to start
	img::copy(src_gray_image, dst_gray_image);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_image, range);
	auto dst_sub = img::sub_view(dst_gray_image, range);
	img::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::blur(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::gradients(src_sub, dst_sub);

	img::write_image(dst_gray_image, out_dir / "combo.png");
}


void gradient_times(fs::path const& out_dir)
{
	std::cout << "\ngradients:\n";

	u32 n_image_sizes = 1;
	u32 image_dim_factor = 4;

	u32 n_image_counts = 3;
	u32 image_count_factor = 4;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 10;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::multi_chart_data_t seq_image_times;
	seq_image_times.color = green;

	img::multi_chart_data_t seq_view_times;
	seq_view_times.color = blue;

	img::multi_chart_data_t par_image_times;
	par_image_times.color = green;

	img::multi_chart_data_t par_view_times;
	par_view_times.color = blue;

	img::multi_chart_data_t simd_image_times;
	simd_image_times.color = green;

	img::multi_chart_data_t simd_view_times;
	simd_view_times.color = blue;

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

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;
		std::vector<r32> seq_image;
		std::vector<r32> seq_view;
		std::vector<r32> par_image;
		std::vector<r32> par_view;
		std::vector<r32> simd_image;
		std::vector<r32> simd_view;
		GrayImage src;
		GrayImage dst;
		GrayImage tmp;
		img::make_image(src, width, height);
		img::make_image(dst, width, height);
		img::make_image(tmp, width, height);
		auto src_v = img::make_view(src);
		auto dst_v = img::make_view(dst);

		for (u32 c = 0; c < n_image_counts; ++c)
		{
			print_count();

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::seq::gradients(src, dst, tmp);
			}			
			t = sw.get_time_milli();
			seq_image.push_back(scale(t));
			print_t(" seq image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::seq::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			seq_view.push_back(scale(t));
			print_t("  seq view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			par_image.push_back(scale(t));
			print_t(" par image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			par_view.push_back(scale(t));
			print_t("  par view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::simd::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			simd_image.push_back(scale(t));
			print_t("simd image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::simd::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			simd_view.push_back(scale(t));
			print_t(" simd view");

			image_count *= image_count_factor;
		}

		seq_image_times.data_list.push_back(seq_image);
		seq_view_times.data_list.push_back(seq_view);
		par_image_times.data_list.push_back(par_image);
		par_view_times.data_list.push_back(par_view);
		simd_image_times.data_list.push_back(simd_image);
		simd_view_times.data_list.push_back(simd_view);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}

	img::grouped_multi_chart_data_t chart_data
	{ 
		seq_image_times, seq_view_times,
		par_image_times, par_view_times,
		simd_image_times, simd_view_times
	};
	Image chart;
	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir / "gradients.bmp");
}


void empty_dir(fs::path const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}


void print(ImageView const& view)
{
	auto w = view.width;
	auto h = view.height;

	std::cout << "width: " << w << " height: " << h << "\n";
}


void print(GrayView const& view)
{
	auto w = view.width;
	auto h = view.height;

	std::cout << "width: " << w << " height: " << h << "\n";
}


void print(img::stats_t const& stats)
{
	std::cout << "mean = " << (double)stats.mean << " sigma = " << (double)stats.std_dev << '\n';
}