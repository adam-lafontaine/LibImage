#include "../../libimage_compact/libimage.hpp"

#include "../utils/stopwatch.hpp"

//#define CHECK_LEAKS

#if defined(_WIN32) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

#include <cstdio>

namespace img = libimage;

using Image = img::image_t;
using ImageView = img::view_t;
using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using Pixel = img::pixel_t;

using path_t = fs::path;


// set this directory for your system
constexpr auto ROOT_DIR = "C:/D_Data/repos/LibImage/";

constexpr auto TEST_IMAGE_DIR = "TestImages/";
constexpr auto IMAGE_IN_DIR = "in_files/";
constexpr auto IMAGE_OUT_DIR = "out_files/";

const auto ROOT_PATH = path_t(ROOT_DIR);
const auto TEST_IMAGE_PATH = ROOT_PATH / TEST_IMAGE_DIR;
const auto IMAGE_IN_PATH = TEST_IMAGE_PATH / IMAGE_IN_DIR;
const auto IMAGE_OUT_PATH = TEST_IMAGE_PATH / IMAGE_OUT_DIR;

const auto CORVETTE_PATH = IMAGE_IN_PATH / "corvette.png";
const auto CADILLAC_PATH = IMAGE_IN_PATH / "cadillac.png";
const auto WEED_PATH = IMAGE_IN_PATH / "weed.png";


bool directory_files_test()
{
	auto title = "directory_files_test";
	printf("\n%s:\n", title);

	auto const test_dir = [](path_t const& dir)
	{
		auto result = fs::is_directory(dir);
		auto msg = result ? "PASS" : "FAIL";
		printf("%s: %s\n", dir.string().c_str(), msg);

		return result;
	};

	auto result =
		test_dir(ROOT_PATH) &&
		test_dir(TEST_IMAGE_PATH) &&
		test_dir(IMAGE_IN_PATH) &&
		test_dir(IMAGE_OUT_PATH);

	auto const test_file = [](path_t const& file) 
	{
		auto result = fs::exists(file);
		auto msg = result ? "PASS" : "FAIL";
		printf("%s: %s\n", file.string().c_str(), msg);

		return result;
	};

	result =
		result &&
		test_file(CORVETTE_PATH) &&
		test_file(CADILLAC_PATH) &&
		test_file(WEED_PATH);

	return result;
}


void empty_dir(path_t const& dir);

void read_write_image_test();
void resize_test();
void view_test();
void alpha_blend_test();
void grayscale_test();
void binary_test();


int main()
{
#if defined(_WIN32) && defined(_DEBUG) && defined(CHECK_LEAKS)
	int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
	dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
	dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
	_CrtSetDbgFlag(dbgFlags);
#endif

	if (!directory_files_test())
	{
		return EXIT_FAILURE;
	}

	read_write_image_test();
	resize_test();
	view_test();
	alpha_blend_test();
	grayscale_test();
	binary_test();
}


void read_write_image_test()
{
	auto title = "read_write_image_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	img::write_image(image, out_dir / "corvette.bmp");
	img::write_image(image, out_dir / "corvette.png");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	img::write_image(gray, out_dir / "cadillac_gray.bmp");
	img::write_image(gray, out_dir / "cadillac_gray.png");
}


void resize_test()
{
	auto title = "resize_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image vertical;
	vertical.width = width / 2;
	vertical.height = height * 2;
	img::resize_image(image, vertical);
	img::write_image(vertical, out_dir / "vertical.png");

	Image horizontal;
	horizontal.width = width * 2;
	horizontal.height = height / 2;
	img::resize_image(image, horizontal);
	img::write_image(horizontal, out_dir / "horizontal.png");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	GrayImage vertical_gray;
	vertical_gray.width = width / 2;
	vertical_gray.height = height * 2;
	img::resize_image(gray, vertical_gray);
	img::write_image(vertical_gray, out_dir / "vertical_gray.png");

	GrayImage horizontal_gray;
	horizontal_gray.width = width * 2;
	horizontal_gray.height = height / 2;
	img::resize_image(gray, horizontal_gray);
	img::write_image(horizontal_gray, out_dir / "horizontal_gray.png");		 
}


void view_test()
{
	auto title = "view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	img::pixel_range_t r{};

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view = img::sub_view(image, r);
	img::write_view(view, out_dir / "view.png");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view_gray = img::sub_view(gray, r);
	img::write_view(view_gray, out_dir / "view_gray.png");
}


void alpha_blend_test()
{
	auto title = "alpha_blend_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	Image cur;
	cur.width = width;
	cur.height = height;
	img::resize_image(caddy, cur);

	Image dst;
	img::make_image(dst, width, height);

	img::seq::transform_alpha(src, [](auto const& p) { return 128; });

	img::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "alpha_blend.png");

	img::seq::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "seq_alpha_blend.png");

	/*img::simd::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "simd_alpha_blend.png");*/

	img::seq::copy(cur, dst);
	img::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "alpha_blend_src_dst.png");

	img::seq::copy(cur, dst);
	img::seq::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "seq_alpha_blend_src_dst.png");

	/*img::seq::copy(cur, dst);
	img::simd::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "simd_alpha_blend_src_dst.png");*/

}


void grayscale_test()
{
	auto title = "grayscale_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	GrayImage dst;
	img::make_image(dst, width, height);

	img::grayscale(src, dst);
	img::write_image(dst, out_dir / "grayscale.png");

	img::seq::grayscale(src, dst);
	img::write_image(dst, out_dir / "seq_grayscale.png");

	/*img::simd::grayscale(src, dst);
	img::write_image(dst, out_dir / "simd_grayscale.png");*/
}


void binary_test()
{
	auto title = "binary_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	GrayImage caddy_gray;
	img::make_image(caddy_gray, caddy.width, caddy.height);

	img::grayscale(caddy, caddy_gray);
	img::write_image(caddy_gray, out_dir / "caddy_gray.png");

	GrayImage caddy_binary;
	img::make_image(caddy_binary, caddy.width, caddy.height);

	img::binarize_th(caddy_gray, caddy_binary, 128);
	img::write_image(caddy_binary, out_dir / "caddy_binary.png");

	Image weed;
	img::read_image_from_file(WEED_PATH, weed);
	auto width = weed.width;
	auto height = weed.height;

	GrayImage binary_src;
	img::make_image(binary_src, width, height);

	GrayImage binary_dst;
	img::make_image(binary_dst, width, height);

	GrayImage temp;
	img::make_image(temp, width, height);

	auto const is_white = [](Pixel p)
	{
		return ((r32)p.red + (r32)p.blue + (r32)p.green) / 3.0f < 190;
	};

	img::binarize(weed, binary_src, is_white);
	img::write_image(binary_src, out_dir / "weed.bmp");

	// centroid point
	auto pt = img::centroid(binary_src);

	// region around centroid
	img::pixel_range_t c{};
	c.x_begin = pt.x - 10;
	c.x_end = pt.x + 10;
	c.y_begin = pt.y - 10;
	c.y_end = pt.y + 10;

	// draw binary image with centroid region
	img::copy(binary_src, binary_dst);
	auto c_view = img::sub_view(binary_dst, c);
	std::fill(c_view.begin(), c_view.end(), 0);
	img::write_image(binary_dst, out_dir / "centroid.bmp");

	// thin the object
	img::seq::thin_objects(binary_src, binary_dst);
	img::write_image(binary_dst, out_dir / "thin.bmp");
}


void empty_dir(path_t const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}