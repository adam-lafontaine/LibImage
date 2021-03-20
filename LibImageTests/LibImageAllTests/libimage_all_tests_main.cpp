#include "../../libimage_all/libimage_all.hpp"

//#define CHECK_LEAKS

#if defined(_WIN32) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

#include <iostream>
#include <mutex>

namespace fs = std::filesystem;
namespace img = libimage;

const auto ROOT_PATH = fs::path("D:/repos/LibImage/LibImageTests/StbTests");

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
void for_each_tests(fs::path const& out_dir);
void transform_tests(fs::path const& out_dir);
void math_tests(fs::path const& out_dir);


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

	basic_tests(dst_root);
	for_each_tests(dst_root);
	transform_tests(dst_root);
	math_tests(dst_root);

	std::cout << "\nDone.\n";
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

	std::cout << '\n';
}


void transform_tests(fs::path const& out_dir)
{
	std::cout << "transform_pixels:\n";

	img::image_t image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);
	auto view = img::make_view(image);

	auto const func = [](img::pixel_t const& p) 
	{ 
		auto pix = p;
		pix.red /= 2;
		pix.green /= 2;
		pix.blue /= 2;
		return pix;
	};

	img::image_t dst_image;
	img::make_image(dst_image, image.width, image.height);
	auto dst_view = img::make_view(dst_image);

	img::seq::transform_pixels(view, dst_view, func);
	img::write_view(dst_view, out_dir / "transform_seq.png");

	img::par::transform_pixels(view, dst_view, func);
	img::write_view(dst_view, out_dir / "transform_par.png");

	img::gray::image_t image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	auto view_gray = img::make_view(image_gray);

	auto const func_gray = [](img::gray::pixel_t const& p) 
	{ 
		auto pix = p;
		pix /= 2;
		return pix;
	};

	img::gray::image_t dst_image_gray;
	img::make_image(dst_image_gray, image_gray.width, image_gray.height);
	auto dst_view_gray = img::make_view(dst_image_gray);

	img::seq::transform_pixels(view_gray, dst_view_gray, func_gray);
	img::write_view(dst_view_gray, out_dir / "transform_gray_seq.png");

	img::par::transform_pixels(view_gray, dst_view_gray, func_gray);
	img::write_view(dst_view_gray, out_dir / "transform_gray_par.png");

	std::cout << '\n';
}


void for_each_tests(fs::path const& out_dir)
{
	std::cout << "for_each_pixel:\n";
	using uint_t = unsigned long long;
	std::mutex mtx;

	img::image_t image;
	img::read_image_from_file(SRC_IMAGE_PATH, image);
	auto view = img::make_view(image);

	uint_t total_red = 0;
	auto const count_func = [&](img::pixel_t const& p) { total_red += p.red; };
	img::seq::for_each_pixel(view, count_func);
	std::cout << "red seq: " << total_red << "\n";

	total_red = 0;
	auto const lk_count_func = [&](img::pixel_t const& p) { std::lock_guard<std::mutex> lk(mtx); total_red += p.red; };
	img::par::for_each_pixel(view, lk_count_func);
	std::cout << "red par: " << total_red << "\n";

	img::gray::image_t image_gray;
	img::read_image_from_file(SRC_IMAGE_PATH, image_gray);
	auto view_gray = img::make_view(image_gray);

	uint_t total_gray = 0;
	auto const count_func_gray = [&](img::gray::pixel_t const& p) { total_gray += p; };
	img::seq::for_each_pixel(view_gray, count_func_gray);
	std::cout << "gray seq: " << total_gray << "\n";

	total_gray = 0;
	auto const lk_count_func_gray = [&](img::gray::pixel_t const& p) { std::lock_guard<std::mutex> lk(mtx); total_gray += p; };
	img::par::for_each_pixel(view_gray, lk_count_func_gray);
	std::cout << "gray par: " << total_gray << "\n";


	auto const red_func = [](img::pixel_t& p) { p.red /= 2; };
	img::seq::for_each_pixel(view, red_func);
	img::write_view(view, out_dir / "for_each_seq.png");
	img::par::for_each_pixel(view, red_func);
	img::write_view(view, out_dir / "for_each_par.png");


	auto const gray_func = [](img::gray::pixel_t& p) { p /= 2; };
	img::seq::for_each_pixel(view_gray, gray_func);
	img::write_view(view_gray, out_dir / "for_each_gray_seq.png");
	img::par::for_each_pixel(view_gray, gray_func);
	img::write_view(view_gray, out_dir / "for_each_gray_par.png");

	std::cout << '\n';
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