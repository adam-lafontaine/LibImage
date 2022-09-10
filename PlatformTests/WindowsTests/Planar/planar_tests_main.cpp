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
void for_each_pixel_test();
void for_each_xy_test();
void grayscale_test();
void select_channel_test();
void alpha_blend_test();
void transform_test();
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
	for_each_pixel_test();
	//for_each_xy_test();
	grayscale_test();
	select_channel_test();
	alpha_blend_test();
	transform_test();
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
	auto title = "read_write_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	write_image(image, "corvette.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	write_image(gray, "cadillac_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
}


void resize_test()
{
	auto title = "resize_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image vertical;
	vertical.width = width / 2;
	vertical.height = height * 2;
	img::resize_image(image, vertical);
	write_image(vertical, "vertical.bmp");

	Image horizontal;
	horizontal.width = width * 2;
	horizontal.height = height / 2;
	img::resize_image(image, horizontal);
	write_image(horizontal, "horizontal.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	GrayImage vertical_gray;
	vertical_gray.width = width / 2;
	vertical_gray.height = height * 2;
	img::resize_image(gray, vertical_gray);
	write_image(vertical_gray, "vertical_gray.bmp");

	GrayImage horizontal_gray;
	horizontal_gray.width = width * 2;
	horizontal_gray.height = height / 2;
	img::resize_image(gray, horizontal_gray);
	write_image(horizontal_gray, "horizontal_gray.bmp");

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
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

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

	write_image(image_dst, "convert4.bmp");

	img::Image3Cr32 image3;
	img::make_image(image3, width, height);

	img::convert(image, image3);
	img::convert(image3, image_dst);

	write_image(image_dst, "convert3.bmp");

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

	write_image(gray_dst, "convert1.bmp");

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
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

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

	write_image(vette, "swap.bmp");

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

	write_image(caddy, "copy.bmp");		

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
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

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
	write_image(image, "fill_01.bmp");

	img::fill(left_view, green);
	write_image(image, "fill_02.bmp");

	img::Image3Cr32 image3;
	img::make_image(image3, width / 2, height / 2);
	img::View3Cr32 view3 = img::make_view(image3);
	img::fill(view3, blue);
	img::convert(view3, top_left_view);
	write_image(image, "fill_03.bmp");

	img::Image4Cr32 image4;	
	img::make_image(image4, width / 2, height / 2);
	img::View4Cr32 view4 = img::make_view(image4);

	img::fill(view4, white);
	img::convert(view4, bottom_right_view);
	write_image(image, "fill_04.bmp");

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
	write_image(image, "fill_view3.bmp");

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
	write_image(image, "fill_view4.bmp");

	GrayImage gray;
	img::make_image(gray, width, height);

	auto gr_top_left_view = img::sub_view(gray, top_left);
	auto gr_bottom_right_view = img::sub_view(gray, bottom_right);

	img::fill(gray, 128);
	write_image(gray, "gray_fill_01.bmp");

	img::fill(gr_top_left_view, 0);
	img::fill(gr_bottom_right_view, 255);
	write_image(gray, "gray_fill_02.bmp");

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
	img::View1Cr32 top_view1 = img::make_view(top1);
	img::convert(gr_top_view, top1);

	img::Image1Cr32 bottom1;
	img::make_image(bottom1, width, view_height);
	img::View1Cr32 bottom_view1 = img::make_view(bottom1);
	img::convert(gr_bottom_view, bottom_view1);

	img::copy(bottom_view1, top_view1);

	img::convert(top1, gr_top_view);

	write_image(gray, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
	img::destroy_image(top1);
	img::destroy_image(bottom1);
}


void for_each_pixel_test()
{
	auto title = "for_each_pixel_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::for_each_pixel(vette, [](u8& p) { p = 255 - p; });
	write_image(vette, "invert.bmp");
	
	Range2Du32 right{};
	right.x_begin = width / 2;
	right.x_end = width;
	right.y_begin = 0;
	right.y_end = height;

	auto view = img::sub_view(vette, right);
	img::for_each_pixel(view, [](u8& p) { p = 255 - p; });
	write_image(vette, "invert_view.bmp");

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	right.x_begin = width / 2;
	right.x_end = width;
	right.y_begin = 0;
	right.y_end = height;

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;	

	img::Image1Cr32 caddy1;
	img::make_image(caddy1, width, height);
	img::convert(caddy, caddy1);

	auto caddy1_left = img::sub_view(caddy1, left);
	auto caddy1_right = img::sub_view(caddy1, right);

	img::for_each_pixel(caddy1_left, [](r32& p) { p = (p < 0.1f) ? 0.0f : (p - 0.1f); });
	img::for_each_pixel(caddy1_right, [](r32& p) { p = (p > 0.9f) ? 1.0f : (p + 0.1f); });

	auto caddy_left = img::sub_view(caddy, left);
	auto caddy_right = img::sub_view(caddy, right);

	img::convert(caddy1_left, caddy_left);
	img::convert(caddy1_right, caddy_right);

	write_image(caddy, "light_dark_1.bmp");

	Image rgba;
	img::read_image_from_file(CORVETTE_PATH, rgba);
	width = rgba.width;
	height = rgba.height;
	
	img::Image3Cr32 rgb;
	img::make_image(rgb, width, height);
	img::View3Cr32 rgb_view = img::make_view(rgb);
	auto red = img::select_channel(rgb_view, img::RGB::R);

	img::convert(rgba, rgb);

	img::for_each_pixel(red, [](r32& p) { p = 1.0f - p; });

	img::convert(rgb, rgba);
	write_image(rgba, "invert_green.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy);
	img::destroy_image(caddy1);
	img::destroy_image(rgba);
	img::destroy_image(rgb);
}


void for_each_xy_test()
{
	auto title = "for_each_xy_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };


}


void grayscale_test()
{
	auto title = "grayscale_test";
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
	img::make_image(image3, width / 2, height);
	img::View3Cr32 view3 = img::make_view(image3);

	img::Image4Cr32 image4;
	img::make_image(image4, width / 2, height);
	img::View4Cr32 view4 = img::make_view(image4);

	img::convert(left_view, image3);
	img::convert(right_view, image4);

	GrayImage dst;
	img::make_image(dst, width, height);

	img::Image1Cr32 image1;
	img::make_image(image1, width, height);

	auto gr_left = img::sub_view(image1, left);
	auto gr_right = img::sub_view(image1, right);

	img::grayscale(view3, gr_right);
	img::grayscale(view4, gr_left);

	img::convert(image1, dst);

	write_image(image, "image.bmp");
	write_image(dst, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(image3);
	img::destroy_image(image4);
	img::destroy_image(dst);
	img::destroy_image(image1);
}


void select_channel_test()
{
	auto title = "select_channel_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	img::Image3Cr32 vette_img3;
	img::make_image(vette_img3, vette.width, vette.height);
	img::View3Cr32 vette3 = img::make_view(vette_img3);
	img::convert(vette, vette3);

	GrayImage vette_dst;
	img::make_image(vette_dst, vette.width, vette.height);

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	img::Image4Cr32 caddy_img4;
	img::make_image(caddy_img4, caddy.width, caddy.height);
	img::View4Cr32 caddy4 = img::make_view(caddy_img4);
	img::convert(caddy, caddy4);

	GrayImage caddy_dst;
	img::make_image(caddy_dst, caddy.width, caddy.height);

	auto red = img::select_channel(vette3, img::RGB::R);
	auto blue = img::select_channel(caddy4, img::RGBA::B);

	img::convert(red, vette_dst);
	write_image(vette_dst, "red.bmp");

	img::convert(blue, caddy_dst);
	write_image(caddy_dst, "blue.bmp");

	img::destroy_image(vette);
	img::destroy_image(vette_img3);
	img::destroy_image(vette_dst);
	img::destroy_image(caddy);
	img::destroy_image(caddy_img4);
	img::destroy_image(caddy_dst);
}


void alpha_blend_test()
{
	auto title = "alpha_blend_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);

	Image caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(caddy_read, caddy);

	img::Image4Cr32 vette_img4;
	img::make_image(vette_img4, width, height);
	img::View4Cr32 vette4 = img::make_view(vette_img4);
	img::convert(vette, vette4);

	auto alpha_view = img::select_channel(vette4, img::RGBA::A);
	img::for_each_pixel(alpha_view, [](r32& p) { p = 0.5f; });

	img::Image4Cr32 caddy_img4;
	img::make_image(caddy_img4, width, height);
	img::View4Cr32 caddy4 = img::make_view(caddy_img4);
	img::convert(caddy, caddy4);

	img::Image3Cr32 dst_img3;
	img::make_image(dst_img3, width, height);
	img::View3Cr32 dst3 = img::make_view(dst_img3);

	img::alpha_blend(vette4, img::make_rgb_view(caddy4), dst3);

	clear_image(vette);
	img::convert(dst3, vette);
	write_image(vette, "blend_01.bmp");

	img::alpha_blend(vette4, img::make_rgb_view(caddy4));

	clear_image(vette);
	img::convert(caddy4, vette);
	write_image(vette, "blend_02.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy_read);
	img::destroy_image(caddy);
	img::destroy_image(vette_img4);
	img::destroy_image(caddy_img4);
	img::destroy_image(dst_img3);
}


void transform_test()
{
	auto title = "transform_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	auto const invert = [](u8 p) { return 255 - p; };
	auto const lut = img::to_lut(invert);

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_read;
	img::read_image_from_file(CADILLAC_PATH, gr_read);

	GrayImage caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(gr_read, caddy);

	GrayImage gr_dst;
	img::make_image(gr_dst, width, height);

	Range2Du32 right{};
	right.x_begin = width / 2;
	right.x_end = width;
	right.y_begin = 0;
	right.y_end = height;

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	auto vette_right = img::sub_view(vette, right);
	auto caddy_left = img::sub_view(caddy, left);
	auto gr_dst_right = img::sub_view(gr_dst, right);
	auto gr_dst_left = img::sub_view(gr_dst, left);

	img::transform(caddy_left, gr_dst_left, lut);
	img::transform(vette_right, gr_dst_right, lut);

	write_image(gr_dst, "transform_gray.bmp");

	img::Image1Cr32 vette1;
	img::make_image(vette1, width, height);
	img::convert(vette, vette1);

	img::Image1Cr32 caddy1;
	img::make_image(caddy1, width, height);
	img::convert(caddy, caddy1);

	img::Image1Cr32 gr_dst1;
	img::make_image(gr_dst1, width, height);

	auto vette1_right = img::sub_view(vette1, right);
	auto caddy1_left = img::sub_view(caddy1, left);
	auto gr_dst1_right = img::sub_view(gr_dst1, right);
	auto gr_dst1_left = img::sub_view(gr_dst1, left);

	auto const invert1 = [](r32 p) { return 1.0f - p; };

	img::transform(caddy1_left, gr_dst1_right, invert1);
	img::transform(vette1_right, gr_dst1_left, invert1);

	img::convert(gr_dst1, gr_dst);

	write_image(gr_dst, "transform_gray1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_read);
	img::destroy_image(caddy);
	img::destroy_image(gr_dst);
	img::destroy_image(vette1);
	img::destroy_image(caddy1);
	img::destroy_image(gr_dst1);
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