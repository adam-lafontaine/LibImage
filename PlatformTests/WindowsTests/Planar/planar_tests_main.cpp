#include "../../../libimage_planar/libimage.hpp"

#include "../utils/stopwatch.hpp"

//#define CHECK_LEAKS

#if defined(_WIN32) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

#include <cstdio>
#include <algorithm>
#include <filesystem>

namespace img = libimage;
namespace fs = std::filesystem;

using Image = img::Image;
using ImageView = img::View;
using GrayImage = img::gray::Image;
using GrayView = img::gray::View;
using Pixel = img::Pixel;

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

const char* to_cstring(path_t const& path)
{
	return path.string().c_str();
}


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
//void clear_image(Image const& img);
//void clear_image(GrayImage const& img);

void read_write_image_test();
void resize_test();
void convert_test();
void view_test();
//void transform_test();
//void copy_test();
//void fill_test();
//void alpha_blend_test();
//void grayscale_test();
//void binary_test();
//void contrast_test();
//void blur_test();
//void gradients_test();
//void edges_test();
//void combo_view_test();
//void rotate_test();


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
	convert_test();
	view_test();
	//transform_test();
	//copy_test();
	//fill_test();
	//alpha_blend_test();
	//grayscale_test();
	//binary_test();
	//contrast_test();
	//blur_test();
	//gradients_test();
	//edges_test();
	//combo_view_test();
	//rotate_test();
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

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	img::write_image(gray, out_dir / "cadillac_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
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
	img::write_image(vertical, out_dir / "vertical.bmp");

	Image horizontal;
	horizontal.width = width * 2;
	horizontal.height = height / 2;
	img::resize_image(image, horizontal);
	img::write_image(horizontal, out_dir / "horizontal.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	GrayImage vertical_gray;
	vertical_gray.width = width / 2;
	vertical_gray.height = height * 2;
	img::resize_image(gray, vertical_gray);
	img::write_image(vertical_gray, out_dir / "vertical_gray.bmp");

	GrayImage horizontal_gray;
	horizontal_gray.width = width * 2;
	horizontal_gray.height = height / 2;
	img::resize_image(gray, horizontal_gray);
	img::write_image(horizontal_gray, out_dir / "horizontal_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(vertical);
	img::destroy_image(horizontal);
	img::destroy_image(gray);
	img::destroy_image(vertical_gray);
	img::destroy_image(horizontal_gray);
}


void convert_test()
{
	auto title = "convert_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image image_dst;
	img::make_image(image_dst, width, height);

	img::Image4Cr32 image4;
	img::make_image(image4, width, height);

	img::convert(image, image4);
	img::convert(image4, image_dst);

	img::write_image(image_dst, out_dir / "convert4.bmp");

	img::Image3Cr32 image3;
	img::make_image(image3, width, height);

	img::convert(image, image3);
	img::convert(image3, image_dst);

	img::write_image(image_dst, out_dir / "convert3.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	GrayImage gray_dst;
	img::make_image(gray_dst, width, height);

	img::Image1Cr32 image1;
	img::make_image(image1, width, height);

	img::convert(gray, image1);
	img::convert(image1, gray_dst);

	img::write_image(gray_dst, out_dir / "convert1.bmp");

	img::destroy_image(image);
	img::destroy_image(image_dst);
	img::destroy_image(image4);
	img::destroy_image(image3);
	img::destroy_image(gray);
	img::destroy_image(gray_dst);
	img::destroy_image(image1);
}


void view_test()
{
	auto title = "view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::Image3Cr32 vette3;
	img::make_image(vette3, width, height);
	img::convert(vette, vette3);

	img::Image4Cr32 vette4;
	img::make_image(vette4, width, height);
	img::convert(vette, vette4);

	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = width / 4;
	r.y_begin = 0;
	r.y_end = height;

	auto dst4 = img::sub_view(vette, r);
	auto sub3 = img::sub_view(vette3, r);	

	r.x_begin = width * 3 / 4;
	r.x_end = width;

	auto dst3 = img::sub_view(vette, r);
	auto sub4 = img::sub_view(vette4, r);

	img::convert(sub3, dst3);
	img::convert(sub4, dst4);

	img::write_image(vette, out_dir / "swap.bmp");


	/*GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;*/

	

	img::destroy_image(vette);
	img::destroy_image(vette3);
	img::destroy_image(vette4);
	//img::destroy_image(gray);
}


void empty_dir(path_t const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}


//void clear_image(Image const& img)
//{
//	constexpr auto black = img::to_pixel(0, 0, 0);
//	img::fill(img, black);
//}
//
//
//void clear_image(GrayImage const& img)
//{
//	img::fill(img, 0);
//}