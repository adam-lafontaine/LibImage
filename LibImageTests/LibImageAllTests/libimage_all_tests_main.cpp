#include "../../libimage_all/libimage.hpp"
#include "../../libimage_all/math/libimage_math.hpp"
#include "../../libimage_all/proc/process.hpp"

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

const auto ROOT_PATH = fs::path("C:/D_Data/repos/LibImage/LibImageTests/LibImageAllTests");

const auto CORVETTE_PATH = ROOT_PATH / "in_files/png/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH / "in_files/png/cadillac.png";

const auto SRC_IMAGE_PATH = CORVETTE_PATH;
const auto DST_IMAGE_ROOT = ROOT_PATH / "out_files";

const auto RED = ROOT_PATH / "in_files/bmp/red.bmp";


void empty_dir(fs::path const& dir);
void print(img::view_t const& view);
void print(img::gray::view_t const& view);
void print(img::stats_t const& stats);

void basic_tests(fs::path const& out_dir);
void math_tests(fs::path const& out_dir);
void for_each_tests();
void transform_tests();
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
	empty_dir(dst_root);

	/*basic_tests(dst_root);
	math_tests(dst_root);

	for_each_tests();
	transform_tests();*/

	process_tests(dst_root);

	std::cout << "\nDone.\n";
}


void basic_tests(fs::path const& out_dir)
{
	std::cout << "basic:\n";

	img::image_t image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);

	img::write_image(image, out_dir / "image.png");
	img::write_image(image, out_dir / "image.bmp");

	img::image_t red_image;
	img::read_image_from_file(RED, red_image);
	auto red_view = img::make_view(red_image);
	img::write_image(red_image, out_dir / "red_image.png");
	img::write_view(red_view, out_dir / "red_view.png");


	auto view = img::make_view(image);
	print(view);

	img::write_view(view, out_dir / "view.png");
	img::write_view(view, out_dir / "view.bmp");

	auto w = view.width;
	auto h = view.height;

	img::pixel_range_t range = { w * 1 / 3, w * 2 / 3, h * 1 / 3, h * 2 / 3 };
	auto sub_view = img::sub_view(view, range);
	print(sub_view);
	img::write_view(sub_view, out_dir / "sub.png");

	auto row_view = img::row_view(view, h / 2);
	print(row_view);
	img::write_view(row_view, out_dir / "row_view.bmp");

	auto col_view = img::column_view(view, w / 2);
	print(col_view);
	img::write_view(col_view, out_dir / "col_view.bmp");

	img::image_t resize_image;
	resize_image.width = w / 4;
	resize_image.height = h / 2;
	auto resize_view = img::make_resized_view(image, resize_image);
	print(resize_view);
	img::write_image(resize_image, out_dir / "resize_image.bmp");
	img::write_view(resize_view, out_dir / "resize_view.bmp");


	img::gray::image_t image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	img::write_image(image_gray, out_dir / "image_gray.bmp");

	auto view_gray = img::make_view(image_gray);
	print(view_gray);
	img::write_view(view_gray, out_dir / "view_gray.bmp");

	auto sub_view_gray = img::sub_view(view_gray, range);
	print(sub_view_gray);
	img::write_view(sub_view_gray, out_dir / "sub_view_gray.png");

	auto row_view_gray = img::row_view(view_gray, view_gray.height / 2);
	print(row_view_gray);
	img::write_view(row_view_gray, out_dir / "row_view_gray.png");

	auto col_view_gray = img::column_view(view_gray, view_gray.width / 2);
	print(col_view_gray);
	img::write_view(col_view_gray, out_dir / "col_view_gray.png");

	img::gray::image_t resize_image_gray;
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

	img::gray::image_t image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	auto view_gray = img::make_view(image_gray);

	auto stats_gray = img::calc_stats(view_gray);

	img::gray::image_t stats_image_gray;
	img::draw_histogram(stats_gray.hist, stats_image_gray);

	print(stats_gray);

	img::write_image(stats_image_gray, out_dir / "stats_image_gray.png");

	img::image_t image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);
	auto view = img::make_view(image);

	auto stats = img::calc_stats(view);

	img::image_t stats_image;
	img::draw_histogram(stats, stats_image);

	print(stats.red);
	print(stats.green);
	print(stats.blue);

	img::write_image(stats_image, out_dir / "stats_image.png");

	

	auto const binarize = [&](u8 p) { return p > stats_gray.mean ? 255 : 0; };
	img::gray::image_t binary;
	img::make_image(binary, image_gray.width, image_gray.height);
	std::transform(image_gray.begin(), image_gray.end(), binary.begin(), binarize);

	img::write_image(binary, out_dir / "binary.png");

	std::cout << '\n';
}


void make_image(img::image_t& image, u32 size)
{
	u32 width = size / 5;
	u32 height = size / width;

	img::make_image(image, width, height);
}


void for_each_pixel(img::image_t& image, std::function<void(img::pixel_t&)> const& func)
{
	u32 size = image.width * image.height;
	for (u32 i = 0; i < size; ++i)
	{
		func(image.data[i]);
	}
}


void for_each_pixel(img::view_t& view, std::function<void(img::pixel_t&)> const& func)
{
	for (u32 y = 0; y < view.height; ++y)
	{
		auto row = view.row_begin(y);
		for (u32 x = 0; x < view.width; ++x)
		{
			func(row[x]);
		}
	}
}


img::pixel_t alpha_blend_linear(img::pixel_t const& src, img::pixel_t const& current)
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


void for_each_tests()
{
	std::cout << "for_each:\n";

	std::random_device rd;
	std::default_random_engine reng(rd());
	std::uniform_int_distribution<int> dist(0, 255);

	auto const random_pixel = [&]() 
	{
		img::pixel_t p;

		for (u32 i = 0; i < 4; ++i)
		{
			p.channels[i] = static_cast<u8>(dist(reng));
		}

		return p;
	};

	auto const random_blended_pixel = [&](img::pixel_t& p) 
	{
		img::pixel_t src = random_pixel();

		p = alpha_blend_linear(src, p);
	};

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size = size_start;
	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		img::image_t image;
		make_image(image, size);

		std::cout << "image_t size = " << size <<'\n';

		sw.start();
		std::for_each(image.begin(), image.end(), random_blended_pixel);
		auto t = sw.get_time_milli();
		std::cout << "    stl: " << 1000 * t / size << '\n';

		sw.start();
		std::for_each(std::execution::par, image.begin(), image.end(), random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "stl par: " << 1000 * t / size << '\n';

		sw.start();
		for_each_pixel(image, random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "   loop: " << 1000 * t / size << "\n\n";
	}

	img::image_t image;
	make_image(image, size);	
	img::pixel_range_t range = {};

	size = size_start;

	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		range.x_end = size / 5;
		range.y_end = size / range.x_end;

		auto view = img::sub_view(image, range);

		auto sz = view.width * view.height;

		std::cout << "view_t size = " << sz << '\n';

		sw.start();
		std::for_each(view.begin(), view.end(), random_blended_pixel);
		auto t = sw.get_time_milli();
		std::cout << "    stl: " << 1000 * t / sz << '\n';

		sw.start();
		std::for_each(std::execution::par, view.begin(), view.end(), random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "stl par: " << 1000 * t / sz << '\n';

		sw.start();
		for_each_pixel(view, random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "   loop: " << 1000 * t / sz << "\n\n";
	}
}


void transform_tests()
{
	std::cout << "transform:\n";

	std::random_device rd;
	std::default_random_engine reng(rd());
	std::uniform_int_distribution<int> dist(0, 255);

	auto const random_pixel = [&]()
	{
		img::pixel_t p;

		for (u32 i = 0; i < 4; ++i)
		{
			p.channels[i] = static_cast<u8>(dist(reng));
		}

		return p;
	};

	auto const random_blended_pixel = [&](img::pixel_t& p)
	{
		img::pixel_t src = random_pixel();

		return alpha_blend_linear(src, p);
	};

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size =size_start;
	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		img::image_t src;
		make_image(src, size);
		img::image_t dst;
		make_image(dst, size);

		std::cout << "image_t size = " << size << '\n';

		sw.start();
		std::transform(src.begin(), src.end(), dst.begin(), random_blended_pixel);
		auto t = sw.get_time_milli();
		std::cout << "    stl: " << 1000 * t / size << '\n';

		sw.start();
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "stl par: " << 1000 * t / size << "\n\n";
	}

	img::image_t src;
	make_image(src, size);
	img::image_t dst;
	make_image(dst, size);
	img::pixel_range_t range = {};

	size = size_start;

	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		range.x_end = size / 5;
		range.y_end = size / range.x_end;

		auto src_view = img::sub_view(src, range);
		auto dst_view = img::sub_view(dst, range);

		auto sz = src_view.width * src_view.height;

		std::cout << "view_t size = " << src_view.width * src_view.height << '\n';

		sw.start();
		std::transform(src_view.begin(), src_view.end(), dst_view.begin(), random_blended_pixel);
		auto t = sw.get_time_milli();
		std::cout << "    stl: " << 1000 * t / sz << '\n';

		sw.start();
		std::transform(std::execution::par, src_view.begin(), src_view.end(), dst_view.begin(), random_blended_pixel);
		t = sw.get_time_milli();
		std::cout << "stl par: " << 1000 * t / sz << "\n\n";
	}
}


void process_tests(fs::path const& out_dir)
{
	std::cout << "process:\n";

	// get image
	img::image_t corvette_image;
	img::read_image_from_file(CORVETTE_PATH, corvette_image);
	auto const width = corvette_image.width;
	auto const height = corvette_image.height;
	auto corvette_view = img::make_view(corvette_image);

	img::image_t dst_image;
	auto dst_view = img::make_view(dst_image, width, height);

	img::gray::image_t dst_gray_image;
	auto dst_gray_view = img::make_view(dst_gray_image, width, height);

	// get another image for blending
	// make sure it is the same size
	img::image_t caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	img::image_t caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(caddy_read, caddy);
	auto caddy_view = make_view(caddy);

	// alpha blending
	img::convert_alpha(caddy_view, [](auto const& p) { return 128; });
	img::alpha_blend(caddy_view, corvette_view, dst_view);
	img::write_image(dst_image, out_dir / "alpha_blend.png");

	// grayscale
	img::convert_grayscale(corvette_view, dst_gray_view);
	img::write_image(dst_gray_image, out_dir / "convert_grayscale.png");
	
	// stats
	auto gray_stats = img::calc_stats(dst_gray_image);
	img::gray::image_t gray_stats_image;
	img::draw_histogram(gray_stats.hist, gray_stats_image);
	img::write_image(gray_stats_image, out_dir / "gray_stats.png");
	print(gray_stats);

	// alpha grayscale
	img::convert_alpha_grayscale(corvette_view);
	auto alpha_stats = img::calc_stats(corvette_image, img::Channel::Alpha);
	img::gray::image_t alpha_stats_image;
	img::draw_histogram(alpha_stats.hist, alpha_stats_image);
	img::write_image(alpha_stats_image, out_dir / "alpha_stats.png");
	print(alpha_stats);

	// create a new grayscale source
	img::gray::image_t src_gray_image;
	auto src_gray_view = img::make_view(src_gray_image, width, height);
	img::copy(dst_gray_view, src_gray_view);

	// contrast
	auto shade_min = static_cast<u8>(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = static_cast<u8>(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));
	img::adjust_contrast(src_gray_view, dst_gray_view, shade_min, shade_max);
	img::write_image(dst_gray_image, out_dir / "contrast.png");

	// binarize
	auto const is_white = [&](u8 p) { return static_cast<r32>(p) > gray_stats.mean; };
	img::binarize(src_gray_view, dst_gray_view, is_white);
	img::write_image(dst_gray_image, out_dir / "binarize.png");

	//blur
	img::blur(src_gray_view, dst_gray_view);
	img::write_image(dst_gray_image, out_dir / "blur.png");
	
	// combine transformations in the same image
	// regular grayscale to start
	img::copy(src_gray_view, dst_gray_view);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_view, range);
	auto dst_sub = img::sub_view(dst_gray_view, range);
	img::adjust_contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_view, range);
	dst_sub = img::sub_view(dst_gray_view, range);
	img::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_view, range);
	dst_sub = img::sub_view(dst_gray_view, range);
	auto const is_black = [&](u8 p) { return static_cast<r32>(p) < gray_stats.mean; };
	img::binarize(src_sub, dst_sub, is_black);

	img::write_image(dst_gray_image, out_dir / "combo.png");

	range.x_begin = width / 2;
	range.x_end = width;

	

}


void empty_dir(fs::path const& dir)
{
	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}


void print(img::view_t const& view)
{
	auto w = view.width;
	auto h = view.height;

	std::cout << "width: " << w << " height: " << h << "\n";
}


void print(img::gray::view_t const& view)
{
	auto w = view.width;
	auto h = view.height;

	std::cout << "width: " << w << " height: " << h << "\n";
}


void print(img::stats_t const& stats)
{
	std::cout << "mean = " << (double)stats.mean << " sigma = " << (double)stats.std_dev << '\n';
}