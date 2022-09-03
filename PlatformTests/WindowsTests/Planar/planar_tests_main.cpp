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
void clear_image(Image const& img);
void clear_image(GrayImage const& img);

void read_write_image_test();
void resize_test();
void convert_test();
void view_test();
void fill_test();
void copy_test();
//void transform_test();


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
	fill_test();
	copy_test();
	//transform_test();
	
	
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

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	img::Image1Cr32 caddy1;
	img::make_image(caddy1, width, height);
	img::convert(caddy, caddy1);

	r.x_begin = 0;
	r.x_end = width / 2;
	r.y_begin = 0;
	r.y_end = height;

	auto sub1 = img::sub_view(caddy1, r);

	r.x_begin = width / 4;
	r.x_end = width * 3 / 4;

	auto dst1 = img::sub_view(caddy, r);

	img::convert(sub1, dst1);

	img::write_image(caddy, out_dir / "copy.bmp");		

	img::destroy_image(vette);
	img::destroy_image(vette3);
	img::destroy_image(vette4);
	img::destroy_image(caddy);
}


void fill_test()
{
	auto title = "fill_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	u32 width = 800;
	u32 height = 800;

	auto const red = img::to_pixel(255, 0, 0);
	auto const green = img::to_pixel(0, 255, 0);
	auto const blue = img::to_pixel(0, 0, 255);
	auto const black = img::to_pixel(0, 0, 0);
	auto const white = img::to_pixel(255, 255, 255);

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	Range2Du32 top_left{};
	top_left.x_begin = 0;
	top_left.x_end = width / 2;
	top_left.y_begin = 0;
	top_left.y_end = height / 2;

	Range2Du32 bottom_right{};
	bottom_right.x_begin = width / 2;
	bottom_right.x_end = width;
	bottom_right.y_begin = height / 2;
	bottom_right.y_end = height;

	Image image;
	img::make_image(image, width, height);

	auto left_view = img::sub_view(image, left);
	auto top_left_view = img::sub_view(image, top_left);
	auto bottom_right_view = img::sub_view(image, bottom_right);

	img::fill(image, red);
	img::write_image(image, out_dir / "fill_01.bmp");

	img::fill(left_view, green);
	img::write_image(image, out_dir / "fill_02.bmp");

	img::Image3Cr32 image3;
	img::make_image(image3, width / 2, height / 2);
	img::fill(image3, blue);
	img::convert(image3, top_left_view);
	img::write_image(image, out_dir / "fill_03.bmp");

	img::Image4Cr32 image4;	
	img::make_image(image4, width / 2, height / 2);
	img::fill(image4, white);
	img::convert(image4, bottom_right_view);
	img::write_image(image, out_dir / "fill_04.bmp");

	img::destroy_image(image3);
	img::destroy_image(image4);

	img::make_image(image3, width, height);
	u32 x_step = width / 5;
	Range2Du32 r{};
	r.y_begin = 0;
	r.y_end = height;
	for (u32 x = 0; x < width; x +=x_step)
	{
		r.x_begin = x;
		r.x_end = x + x_step;

		auto view = img::sub_view(image3, r);
		auto red = (u8)(255.0f * r.x_end / width);
		img::fill(view, img::to_pixel(red, 255, 0));
	}
	img::convert(image3, image);
	img::write_image(image, out_dir / "fill_view3.bmp");

	img::make_image(image4, width, height);
	u32 y_step = height / 5;
	r.x_begin = 0;
	r.x_end = width;
	for (u32 y = 0; y < height; y += y_step)
	{
		r.y_begin = y;
		r.y_end = y + y_step;

		auto view = img::sub_view(image4, r);
		auto blue = (u8)(255.0f * r.y_end / height);
		img::fill(view, img::to_pixel(255, 0, blue));
	}
	img::convert(image4, image);
	img::write_image(image, out_dir / "fill_view4.bmp");

	GrayImage gray;
	img::make_image(gray, width, height);

	auto gr_top_left_view = img::sub_view(gray, top_left);
	auto gr_bottom_right_view = img::sub_view(gray, bottom_right);

	img::fill(gray, 128);
	img::write_image(gray, out_dir / "gray_fill_01.bmp");

	img::fill(gr_top_left_view, 0);
	img::fill(gr_bottom_right_view, 255);
	img::write_image(gray, out_dir / "gray_fill_02.bmp");

	img::destroy_image(image);
	img::destroy_image(image3);
	img::destroy_image(image4);
	img::destroy_image(gray);
}


void copy_test()
{
	auto title = "copy_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	Range2Du32 right{};
	right.x_begin = width / 2;
	right.x_end = width;
	right.y_begin = 0;
	right.y_end = height;

	auto left_view = img::sub_view(image, left);
	auto right_view = img::sub_view(image, right);

	img::Image3Cr32 image3;
	img::make_image(image3, width, height);
	img::convert(image, image3);
	auto left_view3 = img::sub_view(image3, left);
	auto right_view3 = img::sub_view(image3, right);

	img::Image4Cr32 image4;
	img::make_image(image4, width, height);
	img::convert(image, image4);
	auto left_view4 = img::sub_view(image4, left);
	auto right_view4 = img::sub_view(image4, right);

	img::copy(left_view3, right_view3);
	img::copy(right_view4, left_view4);	

	clear_image(image);

	img::convert(right_view3, right_view);
	img::convert(left_view4, left_view);
	write_image(image, "image.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	auto view_height = height / 3;

	Range2Du32 top{};
	top.x_begin = 0;
	top.x_end = width;
	top.y_begin = 0;
	top.y_end = view_height;

	Range2Du32 bottom{};
	bottom.x_begin = 0;
	bottom.x_end = width;
	bottom.y_begin = height - view_height;
	bottom.y_end = height;	

	auto gr_top_view = img::sub_view(gray, top);
	auto gr_bottom_view = img::sub_view(gray, bottom);

	img::Image1Cr32 top1;
	img::make_image(top1, width, view_height);
	img::convert(gr_top_view, top1);

	img::Image1Cr32 bottom1;
	img::make_image(bottom1, width, view_height);
	img::convert(gr_bottom_view, bottom1);

	img::copy(bottom1, top1);

	img::convert(top1, gr_top_view);

	write_image(gray, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
}


void empty_dir(path_t const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}


void clear_image(Image const& img)
{
	constexpr auto black = img::to_pixel(0, 0, 0);
	img::fill(img, black);
}


void clear_image(GrayImage const& img)
{
	img::fill(img, 0);
}