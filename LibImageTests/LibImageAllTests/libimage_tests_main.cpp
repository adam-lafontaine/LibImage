#include "../../libimage/libimage.hpp"
#include "../../libimage/math/libimage_math.hpp"
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
void make_image(Image& image, u32 size);
void make_image(GrayImage& image, u32 size);

void basic_tests(fs::path const& out_dir);
void math_tests(fs::path const& out_dir);
void for_each_tests(fs::path const& out_dir);
void transform_tests(fs::path const& out_dir);
void process_tests(fs::path const& out_dir);


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

	basic_tests(dst_root / "basic");
	math_tests(dst_root / "math");

	for_each_tests(dst_root / "for_each");
	transform_tests(dst_root / "transform");

	process_tests(dst_root / "process");

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
	std::cout << "for_each:\n";
	empty_dir(out_dir);

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
		Pixel src = random_pixel();

		p = alpha_blend_linear(src, p);
	};

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::data_color_t image_loop_times;
	image_loop_times.color = green;

	img::data_color_t image_stl_times;
	image_stl_times.color = green;

	img::data_color_t image_par_times;
	image_par_times.color = green;

	img::data_color_t view_loop_times;
	view_loop_times.color = blue;

	img::data_color_t view_stl_times;
	view_stl_times.color = blue;

	img::data_color_t view_par_times;
	view_par_times.color = blue;

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size = size_start;
	auto const scale = [&](auto t) { return static_cast<r32>(10000 * t / size); };
		

	// compare processing times for views
	size = size_start;
	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		Image image;
		make_image(image, size);

		sw.start();
		img::for_each_pixel(image, random_blended_pixel);
		auto t = sw.get_time_milli();
		image_loop_times.data.push_back(scale(t));

		sw.start();
		std::for_each(image.begin(), image.end(), random_blended_pixel);
		t = sw.get_time_milli();
		image_stl_times.data.push_back(scale(t));

		sw.start();
		std::for_each(std::execution::par, image.begin(), image.end(), random_blended_pixel);
		t = sw.get_time_milli();
		image_par_times.data.push_back(scale(t));

		auto view = img::make_view(image);

		sw.start();
		img::for_each_pixel(view, random_blended_pixel);
		t = sw.get_time_milli();
		view_loop_times.data.push_back(scale(t));

		sw.start();
		std::for_each(view.begin(), view.end(), random_blended_pixel);
		t = sw.get_time_milli();
		view_stl_times.data.push_back(scale(t));

		sw.start();
		std::for_each(std::execution::par, view.begin(), view.end(), random_blended_pixel);
		t = sw.get_time_milli();
		view_par_times.data.push_back(scale(t));
	}

	Image view_chart;
	std::vector<img::data_color_t> view_data = 
	{ 
		image_loop_times, image_stl_times, image_par_times,
		view_loop_times, view_stl_times, view_par_times
	};

	img::draw_bar_chart(view_data, view_chart);
	img::write_image(view_chart, out_dir / "for_each_image_view_times.png");
}


void transform_tests(fs::path const& out_dir)
{
	std::cout << "transform:\n";
	empty_dir(out_dir);

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
		Pixel src = random_pixel();

		return alpha_blend_linear(src, p);
	};

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::data_color_t image_stl_times;
	image_stl_times.color = green;

	img::data_color_t image_par_times;
	image_par_times.color = green;

	img::data_color_t view_stl_times;
	view_stl_times.color = blue;

	img::data_color_t view_par_times;
	view_par_times.color = blue;

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size = size_start;
	auto const scale = [&](auto t) { return static_cast<r32>(10000 * t / size); };

	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		Image image;
		make_image(image, size);
		Image dst;
		make_image(dst, size);

		sw.start();
		std::transform(image.begin(), image.end(), dst.begin(), random_blended_pixel);
		auto t = sw.get_time_milli();
		image_stl_times.data.push_back(scale(t));

		sw.start();
		std::transform(std::execution::par, image.begin(), image.end(), dst.begin(), random_blended_pixel);
		t = sw.get_time_milli();
		image_par_times.data.push_back(scale(t));

		auto view = img::make_view(image);
		auto dst_view = img::make_view(dst);

		sw.start();
		std::transform(view.begin(), view.end(), dst_view.begin(), random_blended_pixel);
		t = sw.get_time_milli();
		view_stl_times.data.push_back(scale(t));

		sw.start();
		std::transform(std::execution::par, view.begin(), view.end(), dst_view.begin(), random_blended_pixel);
		t = sw.get_time_milli();
		view_par_times.data.push_back(scale(t));
	}

	Image view_chart;
	std::vector<img::data_color_t> view_data =
	{
		image_stl_times, image_par_times,
		view_stl_times, view_par_times
	};

	img::draw_bar_chart(view_data, view_chart);
	img::write_image(view_chart, out_dir / "transform_image_view_times.png");
}


void process_tests(fs::path const& out_dir)
{
	std::cout << "process:\n";
	empty_dir(out_dir);

	// get image
	Image corvette_image;
	img::read_image_from_file(CORVETTE_PATH, corvette_image);
	auto const width = corvette_image.width;
	auto const height = corvette_image.height;
	auto corvette_view = img::make_view(corvette_image);

	Image dst_image;
	auto dst_view = img::make_view(dst_image, width, height);

	GrayImage dst_gray_image;
	auto dst_gray_view = img::make_view(dst_gray_image, width, height);

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
	img::par::transform_alpha(caddy_view, [](auto const& p) { return 128; });
	img::par::alpha_blend(caddy_view, corvette_view, dst_view);
	img::write_image(dst_image, out_dir / "alpha_blend.png");

	img::copy(corvette_view, dst_view);
	img::par::alpha_blend(caddy_view, dst_view);
	img::write_image(dst_image, out_dir / "alpha_blend_src_dst.png");

	// grayscale
	img::par::transform_grayscale(corvette_view, dst_gray_view);
	img::write_image(dst_gray_image, out_dir / "convert_grayscale.png");
	
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
	auto src_gray_view = img::make_view(src_gray_image, width, height);
	img::par::copy(dst_gray_view, src_gray_view);

	// contrast
	auto shade_min = static_cast<u8>(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = static_cast<u8>(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));
	img::par::transform_contrast(src_gray_view, dst_gray_view, shade_min, shade_max);
	img::write_image(dst_gray_image, out_dir / "contrast.png");

	// binarize
	auto const is_white = [&](u8 p) { return static_cast<r32>(p) > gray_stats.mean; };
	//img::binarize(src_gray_view, dst_gray_view, is_white);
	img::par::binarize(src_gray_view, dst_gray_view, is_white);
	img::write_image(dst_gray_image, out_dir / "binarize.png");

	//blur
	img::par::blur(src_gray_view, dst_gray_view);
	img::write_image(dst_gray_image, out_dir / "blur.png");
	
	// combine transformations in the same image
	// regular grayscale to start
	img::par::copy(src_gray_view, dst_gray_view);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_view, range);
	auto dst_sub = img::sub_view(dst_gray_view, range);
	img::par::transform_contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_view, range);
	dst_sub = img::sub_view(dst_gray_view, range);
	img::par::binarize(src_sub, dst_sub, is_white);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_view, range);
	dst_sub = img::sub_view(dst_gray_view, range);
	auto const is_black = [&](u8 p) { return static_cast<r32>(p) < gray_stats.mean; };
	img::par::binarize(src_sub, dst_sub, is_black);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_view, range);
	dst_sub = img::sub_view(dst_gray_view, range);
	img::par::blur(src_sub, dst_sub);
	
	img::write_image(dst_gray_image, out_dir / "combo.png");

	// edge detection
	/*GrayImage contrast_gray;
	auto contrast_gray_view = img::make_view(contrast_gray, width, height);
	img::transform_contrast(src_gray_view, contrast_gray_view, shade_min, shade_max);
	img::edges(contrast_gray_view, dst_gray_view, 100);*/
	img::par::edges(src_gray_view, dst_gray_view, 150);
	img::write_image(dst_gray_image, out_dir / "edges.png");

	// gradient
	img::par::gradient(src_gray_view, dst_gray_view);
	img::write_image(dst_gray_image, out_dir / "gradient.png");

	// compare edge detection speeds
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
		img::edges(view, dst_view, 150);
		auto t = sw.get_time_milli();
		seq_times.data.push_back(scale(t));

		sw.start();
		img::par::edges(view, dst_view, 150);
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


void make_image(Image& image, u32 size)
{
	u32 width = size / 5;
	u32 height = size / width;

	img::make_image(image, width, height);
}


void make_image(GrayImage& image, u32 size)
{
	u32 width = size / 5;
	u32 height = size / width;

	img::make_image(image, width, height);
}