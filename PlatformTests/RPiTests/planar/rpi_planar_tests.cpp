#include "../../../libimage_planar/libimage.hpp"

#include "../utils/stopwatch.hpp"

#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <locale.h>

namespace img = libimage;
namespace fs = std::filesystem;

using Image = img::Image;
using ImageView = img::View;
using GrayImage = img::gray::Image;
using GrayView = img::gray::View;
using Pixel = img::Pixel;

using path_t = fs::path;


// set this directory for your system
constexpr auto ROOT_DIR = "../../../";

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
const auto CHESS_PATH = IMAGE_IN_PATH / "chess_board.bmp";


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
		test_file(WEED_PATH) &&
		test_file(CHESS_PATH);

	return result;
}


void empty_dir(path_t const& dir);
void clear_image(Image const& img);
void clear_image(GrayImage const& img);

void read_write_image_test();
void resize_test();
void map_test();
void map_rgb_test();
void map_hsv_test();
void sub_view_test();
void fill_test();
void copy_test();
void for_each_pixel_test();
void for_each_xy_test();
void grayscale_test();
void select_channel_test();
void alpha_blend_test();
void transform_test();
void threshold_test();
void contrast_test();
void blur_test();
void gradients_test();
void edges_test();
void corners_test();
void rotate_test();
void overlay_test();
void scale_down_test();


int main()
{
	Stopwatch sw;
	sw.start();

	if (!directory_files_test())
	{
		return EXIT_FAILURE;
	}

	read_write_image_test();
	resize_test();
	map_test();
	map_rgb_test();
	map_hsv_test();
	sub_view_test();
	fill_test();
	copy_test();
	for_each_pixel_test();
	for_each_xy_test();
	grayscale_test();
	select_channel_test();
	alpha_blend_test();
	transform_test();
	threshold_test();
	contrast_test();
	blur_test();
	gradients_test();
	edges_test();
	corners_test();
	rotate_test();
	overlay_test();
	scale_down_test();

	auto time = sw.get_time_milli();

	auto old_locale = setlocale(LC_NUMERIC, NULL);
	setlocale(LC_NUMERIC, "");

	printf("\nTests complete. %'.3f ms\n", time);

	setlocale(LC_NUMERIC, old_locale);
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


void map_test()
{
	auto title = "map_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	auto width = gray.width;
	auto height = gray.height;

	GrayImage gray_dst;
	img::make_image(gray_dst, width, height);

	img::Buffer32 buffer(width * height);

	img::View1r32 view1;
	img::make_view(view1, width, height, buffer);

	img::map(gray, view1);
	img::map(view1, gray_dst);

	write_image(gray_dst, "map1.bmp");
	
	img::destroy_image(gray);
	img::destroy_image(gray_dst);
	buffer.free();
}


void map_rgb_test()
{
	auto title = "map_rgb_test";
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

	img::Buffer32 buffer(width * height * 4);

	img::View4r32 view4;
	img::make_view(view4, width, height, buffer);

	img::map_rgb(image, view4);
	img::map_rgb(view4, image_dst);
	buffer.reset();

	write_image(image_dst, "map_rgba.bmp");

	img::View3r32 view3;
	img::make_view(view3, width, height, buffer);

	img::map_rgb(image, view3);
	img::map_rgb(view3, image_dst);
	buffer.reset();

	write_image(image_dst, "map_rgb.bmp");

	img::destroy_image(image);
	img::destroy_image(image_dst);
	buffer.free();
}


void map_hsv_test()
{
	auto title = "map_hsv_test";
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

	img::Buffer32 buffer(width * height * 3);

	img::ViewHSVr32 view3;
	img::make_view(view3, width, height, buffer);

	img::map_hsv(image, view3);
	img::map_hsv(view3, image_dst);

	write_image(image_dst, "map_hsv.bmp");

	img::destroy_image(image);
	img::destroy_image(image_dst);
	buffer.free();
}


void sub_view_test()
{
	auto title = "sub_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;
	
	img::Buffer32 buffer(width * height * 7);

	img::View3r32 vette3;
	img::make_view(vette3, width, height, buffer);
	img::map_rgb(vette, vette3);

	img::View4r32 vette4;
	img::make_view(vette4, width, height, buffer);
	img::map_rgb(vette, vette4);

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

	img::map_rgb(sub3, dst3);
	img::map_rgb(sub4, dst4);
	buffer.reset();

	write_image(vette, "swap.bmp");

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	img::View1r32 caddy1;
	img::make_view(caddy1, width, height, buffer);
	img::map(caddy, caddy1);

	r.x_begin = 0;
	r.x_end = width / 2;
	r.y_begin = 0;
	r.y_end = height;

	auto sub1 = img::sub_view(caddy1, r);

	r.x_begin = width / 4;
	r.x_end = width * 3 / 4;

	auto dst1 = img::sub_view(caddy, r);

	img::map(sub1, dst1);

	write_image(caddy, "copy.bmp");		

	img::destroy_image(vette);
	img::destroy_image(caddy);
	buffer.free();
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
	//auto const black = img::to_pixel(0, 0, 0);
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

	GrayImage gray;
	img::make_image(gray, width, height);

	auto gr_top_left_view = img::sub_view(gray, top_left);
	auto gr_bottom_right_view = img::sub_view(gray, bottom_right);

	img::fill(gray, 128);
	write_image(gray, "gray_fill_01.bmp");

	img::fill(gr_top_left_view, 0);
	img::fill(gr_bottom_right_view, 255);
	write_image(gray, "gray_fill_02.bmp");
	
	img::Buffer32 buffer(width * height * 4);
	
	img::View3r32 view3;
	img::make_view(view3, width / 2, height / 2, buffer);
	img::fill(view3, blue);
	img::map_rgb(view3, top_left_view);
	buffer.reset();
	write_image(image, "fill_03.bmp");

	img::View4r32 view4;
	img::make_view(view4, width / 2, height / 2, buffer);

	img::fill(view4, white);
	img::map_rgb(view4, bottom_right_view);
	buffer.reset();
	write_image(image, "fill_04.bmp");


	img::make_view(view3, width, height, buffer);
	u32 x_step = width / 5;
	Range2Du32 r{};
	r.y_begin = 0;
	r.y_end = height;
	for (u32 x = 0; x < width; x +=x_step)
	{
		r.x_begin = x;
		r.x_end = x + x_step;

		auto view = img::sub_view(view3, r);
		auto red = (u8)(255.0f * r.x_end / width);
		img::fill(view, img::to_pixel(red, 255, 0));
	}
	img::map_rgb(view3, image);
	buffer.reset();
	write_image(image, "fill_view3.bmp");

	img::make_view(view4, width, height, buffer);
	u32 y_step = height / 5;
	r.x_begin = 0;
	r.x_end = width;
	for (u32 y = 0; y < height; y += y_step)
	{
		r.y_begin = y;
		r.y_end = y + y_step;

		auto view = img::sub_view(view4, r);
		auto blue = (u8)(255.0f * r.y_end / height);
		img::fill(view, img::to_pixel(255, 0, blue));
	}
	img::map_rgb(view4, image);
	buffer.reset();
	write_image(image, "fill_view4.bmp");	

	img::destroy_image(image);
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
	
	img::Buffer32 buffer(width * height * 7);

	img::View3r32 view3;
	img::make_view(view3, width, height, buffer);
	img::map_rgb(image, view3);
	auto left_view3 = img::sub_view(view3, left);
	auto right_view3 = img::sub_view(view3, right);

	img::View4r32 view4;
	img::make_view(view4, width, height, buffer);
	img::map_rgb(image, view4);
	auto left_view4 = img::sub_view(view4, left);
	auto right_view4 = img::sub_view(view4, right);

	img::copy(left_view3, right_view3);
	img::copy(right_view4, left_view4);	

	clear_image(image);

	img::map_rgb(right_view3, right_view);
	img::map_rgb(left_view4, left_view);
	write_image(image, "image.bmp");

	buffer.reset();

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

	img::View1r32 top1;
	img::make_view(top1, width, view_height, buffer);
	img::map(gr_top_view, top1);

	img::View1r32 bottom1;
	img::make_view(bottom1, width, view_height, buffer);
	img::map(gr_bottom_view, bottom1);

	img::copy(bottom1, top1);

	img::map(top1, gr_top_view);

	write_image(gray, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
	buffer.free();
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

	
	img::Buffer32 buffer(width * height * 4);

	img::View1r32 caddy1;
	img::make_view(caddy1, width, height, buffer);
	img::map(caddy, caddy1);

	auto caddy1_left = img::sub_view(caddy1, left);
	auto caddy1_right = img::sub_view(caddy1, right);

	img::for_each_pixel(caddy1_left, [](r32& p) { p = (p < 0.1f) ? 0.0f : (p - 0.1f); });
	img::for_each_pixel(caddy1_right, [](r32& p) { p = (p > 0.9f) ? 1.0f : (p + 0.1f); });

	auto caddy_left = img::sub_view(caddy, left);
	auto caddy_right = img::sub_view(caddy, right);

	img::map(caddy1_left, caddy_left);
	img::map(caddy1_right, caddy_right);
	buffer.reset();

	write_image(caddy, "light_dark_1.bmp");

	Image rgba;
	img::read_image_from_file(CORVETTE_PATH, rgba);
	width = rgba.width;
	height = rgba.height;
	
	img::View3r32 rgb;
	img::make_view(rgb, width, height, buffer);
	auto red = img::select_channel(rgb, img::RGB::R);

	img::map_rgb(rgba, rgb);

	img::for_each_pixel(red, [](r32& p) { p = 1.0f - p; });

	img::map_rgb(rgb, rgba);
	write_image(rgba, "invert_green.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy);
	img::destroy_image(rgba);
	buffer.free();
}


void for_each_xy_test()
{
	auto title = "for_each_xy_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	u32 width = 800;
	u32 height = 600;

	img::Buffer32 buffer(width * height * 3);

	img::View3r32 view3;
	img::make_view(view3, width, height, buffer);

	auto const xy_func = [&](u32 x, u32 y) 
	{
		auto p = img::rgb_xy_at(view3, x, y);
		p.red() = (r32)x / width;
		p.green() = (r32)y / height;
		p.blue() = (r32)x * y / (width * height);
	};

	img::for_each_xy(view3, xy_func);

	Image image;
	img::make_image(image, width, height);

	img::map_rgb(view3, image);

	write_image(image, "for_each_xy.bmp");

	buffer.free();
	img::destroy_image(image);
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
	
	img::Buffer32 buffer(width * height * 5);

	img::View3r32 view3;
	img::make_view(view3, width / 2, height, buffer);

	img::View4r32 view4;
	img::make_view(view4, width / 2, height, buffer);

	img::map_rgb(left_view, view3);
	img::map_rgb(right_view, view4);

	GrayImage dst;
	img::make_image(dst, width, height);

	img::View1r32 view1;
	img::make_view(view1, width, height, buffer);

	auto gr_left = img::sub_view(view1, left);
	auto gr_right = img::sub_view(view1, right);

	img::grayscale(view3, gr_right);
	img::grayscale(img::make_rgb_view(view4), gr_left);

	img::map(view1, dst);

	write_image(image, "image.bmp");
	write_image(dst, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(dst);
	buffer.free();
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
	auto width = vette.width;
	auto height = vette.height;
	
	img::Buffer32 buffer(width * height * 7);

	img::View3r32 vette3;
	img::make_view(vette3, width, height, buffer);
	img::map_rgb(vette, vette3);

	GrayImage vette_dst;
	img::make_image(vette_dst, vette.width, vette.height);

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	img::View4r32 caddy4;
	img::make_view(caddy4, caddy.width, caddy.height, buffer);
	img::map_rgb(caddy, caddy4);

	GrayImage caddy_dst;
	img::make_image(caddy_dst, caddy.width, caddy.height);

	auto red = img::select_channel(vette3, img::RGB::R);
	auto blue = img::select_channel(caddy4, img::RGBA::B);

	img::map(red, vette_dst);
	write_image(vette_dst, "red.bmp");

	img::map(blue, caddy_dst);
	write_image(caddy_dst, "blue.bmp");

	auto const to_half = [](r32& p) { p *= 0.5f; };

	Image image_dst;
	img::make_image(image_dst, width, height);

	img::map_rgb(vette, vette3);
	auto view_ch = img::select_channel(vette3, img::RGB::R);
	img::for_each_pixel(view_ch, to_half);
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "reduce_r.bmp");

	img::map_rgb(vette, vette3);
	view_ch = img::select_channel(vette3, img::RGB::G);
	img::for_each_pixel(view_ch, to_half);
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "reduce_g.bmp");

	img::map_rgb(vette, vette3);
	view_ch = img::select_channel(vette3, img::RGB::B);
	img::for_each_pixel(view_ch, to_half);
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "reduce_b.bmp");

	img::map_hsv(vette, vette3);
	view_ch = img::select_channel(vette3, img::HSV::H);
	img::for_each_pixel(view_ch, [](r32& p) { p = 0.5f; });
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "change_h.bmp");

	img::map_hsv(vette, vette3);
	view_ch = img::select_channel(vette3, img::HSV::S);
	img::for_each_pixel(view_ch, to_half);
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "reduce_s.bmp");

	img::map_hsv(vette, vette3);
	view_ch = img::select_channel(vette3, img::HSV::V);
	img::for_each_pixel(view_ch, to_half);
	img::map_hsv(vette3, image_dst);
	write_image(image_dst, "reduce_v.bmp");

	img::destroy_image(vette);
	img::destroy_image(vette_dst);
	img::destroy_image(caddy);
	img::destroy_image(caddy_dst);
	img::destroy_image(image_dst);
	buffer.free();
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

	GrayImage gr_vette;
	img::make_image(gr_vette, width, height);
	GrayImage gr_caddy;
	img::make_image(gr_caddy, width, height);

	img::grayscale(vette, gr_vette);
	img::grayscale(caddy, gr_caddy);
	
	img::Buffer32 buffer(width * height * 11);

	img::View4r32 vette4;
	img::make_view(vette4, width, height, buffer);
	img::map_rgb(vette, vette4);

	auto alpha_view = img::select_channel(vette4, img::RGBA::A);
	img::for_each_pixel(alpha_view, [](r32& p) { p = 0.5f; });

	img::View4r32 caddy4;
	img::make_view(caddy4, width, height, buffer);
	img::map_rgb(caddy, caddy4);

	img::View3r32 dst3;
	img::make_view(dst3, width, height, buffer);

	img::alpha_blend(vette4, img::make_rgb_view(caddy4), dst3);

	clear_image(vette);
	img::map_rgb(dst3, vette);
	write_image(vette, "blend_01.bmp");

	img::alpha_blend(vette4, img::make_rgb_view(caddy4), img::make_rgb_view(caddy4));

	clear_image(vette);
	img::map_rgb(caddy4, vette);
	write_image(vette, "blend_02.bmp");

	buffer.reset();

	img::View2r32 caddy2;
	img::make_view(caddy2, width, height, buffer);
	img::map(gr_caddy, img::select_channel(caddy2, img::GA::G));
	alpha_view = img::select_channel(caddy2, img::GA::A);
	img::for_each_pixel(alpha_view, [](r32& p) { p = 0.5f; });

	img::View1r32 vette1;
	img::make_view(vette1, width, height, buffer);
	img::map(gr_vette, vette1);

	img::View1r32 dst1;
	img::make_view(dst1, width, height, buffer);

	img::alpha_blend(caddy2, vette1, dst1);

	img::map(dst1, gr_vette);
	write_image(gr_vette, "gr_blend.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy_read);
	img::destroy_image(caddy);
	img::destroy_image(gr_vette);
	img::destroy_image(gr_caddy);
	buffer.free();
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
	
	img::Buffer32 buffer(width * height * 3);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, buffer);
	img::map(vette, vette1);

	img::View1r32 caddy1;
	img::make_view(caddy1, width, height, buffer);
	img::map(caddy, caddy1);

	img::View1r32 gr_dst1;
	img::make_view(gr_dst1, width, height, buffer);

	auto vette1_right = img::sub_view(vette1, right);
	auto caddy1_left = img::sub_view(caddy1, left);
	auto gr_dst1_right = img::sub_view(gr_dst1, right);
	auto gr_dst1_left = img::sub_view(gr_dst1, left);

	auto const invert1 = [](r32 p) { return 1.0f - p; };

	img::transform(caddy1_left, gr_dst1_right, invert1);
	img::transform(vette1_right, gr_dst1_left, invert1);

	img::map(gr_dst1, gr_dst);

	write_image(gr_dst, "transform_gray1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_read);
	img::destroy_image(caddy);
	img::destroy_image(gr_dst);
	buffer.free();
}


void threshold_test()
{
	auto title = "threshold_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_dst;
	img::make_image(gr_dst, width, height);

	img::threshold(vette, gr_dst, 127, 255);

	write_image(gr_dst, "threshold.bmp");
	
	img::Buffer32 buffer(width * height);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, buffer);
	img::map(vette, vette1);

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	auto vette_left = img::sub_view(vette1, left);

	img::threshold(vette_left, vette_left, 0.0f, 0.5f);
	img::map(vette1, gr_dst);
	write_image(gr_dst, "threshold1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_dst);
	buffer.free();
}


void contrast_test()
{
	auto title = "contrast_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_dst;
	img::make_image(gr_dst, width, height);

	img::contrast(vette, gr_dst, 10, 175);

	write_image(gr_dst, "contrast.bmp");
	
	img::Buffer32 buffer(width * height);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, buffer);
	img::map(vette, vette1);

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	auto vette_left = img::sub_view(vette1, left);

	img::contrast(vette_left, vette_left, 0.1f, 0.75f);
	img::map(vette1, gr_dst);
	write_image(gr_dst, "contrast1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_dst);
	buffer.free();
}


void blur_test()
{
	auto title = "blur_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::Buffer32 buffer(width * height * 6);

	img::View1r32 src;
	img::make_view(src, width, height, buffer);

	img::View1r32 dst;
	img::make_view(dst, width, height, buffer);

	img::map(vette, src);

	img::blur(src, dst);

	img::map(dst, vette);

	write_image(vette, "blur1.bmp");

	buffer.reset();

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	img::View3r32 src3;
	img::make_view(src3, width, height, buffer);

	img::map_rgb(caddy, src3);

	img::View3r32 dst3;
	img::make_view(dst3, width, height, buffer);

	img::blur(src3, dst3);

	img::map_rgb(dst3, caddy);
	write_image(caddy, "blur3.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy);
	buffer.free();
}


void gradients_test()
{
	auto title = "gradients_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::Buffer32 buffer(width * height * 4);

	img::View1r32 src;
	img::make_view(src, width, height, buffer);

	img::View1r32 dst;
	img::make_view(dst, width, height, buffer);

	img::map(vette, src);

	img::gradients(src, dst);

	img::map(dst, vette);
	write_image(vette, "gradients.bmp");

	img::View2r32 xy_dst;
	img::make_view(xy_dst, width, height, buffer);

	img::gradients_xy(src, xy_dst);

	img::map(img::select_channel(xy_dst, img::XY::X), vette, -1.0f, 1.0f);
	write_image(vette, "gradients_x.bmp");

	img::map(img::select_channel(xy_dst, img::XY::Y), vette, -1.0f, 1.0f);
	write_image(vette, "gradients_y.bmp");

	img::destroy_image(vette);
	buffer.free();
}


void edges_test()
{
	auto title = "edges_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::Buffer32 buffer(width * height * 4);

	img::View1r32 src;
	img::make_view(src, width, height, buffer);

	img::View1r32 dst;
	img::make_view(dst, width, height, buffer);

	img::map(vette, src);

	img::edges(src, dst, 0.2f);

	img::map(dst, vette);

	write_image(vette, "edges.bmp");

	img::View2r32 xy_dst;
	img::make_view(xy_dst, width, height, buffer);

	img::edges_xy(src, xy_dst, 0.2f);

	img::map(img::select_channel(xy_dst, img::XY::X), vette);
	write_image(vette, "edges_x.bmp");

	img::map(img::select_channel(xy_dst, img::XY::Y), vette);
	write_image(vette, "edges_y.bmp");

	img::destroy_image(vette);
	buffer.free();
}


void corners_test()
{
	auto title = "corners_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage chess;
	img::read_image_from_file(CHESS_PATH, chess);
	auto width = chess.width;
	auto height = chess.height;

	img::Buffer32 buffer(width * height * 4);

	img::View1r32 src1;
	img::make_view(src1, width, height, buffer);

	img::View1r32 dst1;
	img::make_view(dst1, width, height, buffer);

	img::View2r32 temp2;
	img::make_view(temp2, width, height, buffer);

	img::map(chess, src1);

	img::corners(src1, temp2, dst1);

	img::map(dst1, chess);
	write_image(chess, "corners.bmp");




	img::destroy_image(chess);
	buffer.free();
}


void rotate_test()
{
	auto title = "rotate_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	img::Buffer32 buffer(width * height * 6);

	img::View1r32 src;
	img::make_view(src, width, height, buffer);

	img::View1r32 dst;
	img::make_view(dst, width, height, buffer);

	img::map(vette, src);

	Point2Du32 origin = { width / 2, height / 2 };
	r32 theta = 0.6f * 2 * 3.14159f;

	img::rotate(src, dst, origin, theta);
	
	img::map(dst, vette);
	write_image(vette, "rotate1.bmp");

	buffer.reset();

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	img::View3r32 src3;
	img::make_view(src3, width, height, buffer);

	img::map_rgb(caddy, src3);

	img::View3r32 dst3;
	img::make_view(dst3, width, height, buffer);

	img::rotate(src3, dst3, origin, theta);

	img::map_rgb(dst3, caddy);

	write_image(caddy, "rotate3.bmp");

	img::destroy_image(vette);
	img::destroy_image(caddy);
	buffer.free();
}


void overlay_test()
{
	auto title = "overlay_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_caddy;
	img::read_image_from_file(CADILLAC_PATH, gr_caddy);

	img::Buffer32 buffer(width * height * 5);

	img::View3r32 view3;
	img::make_view(view3, width, height, buffer);
	img::map_rgb(vette, view3);

	img::View1r32 view1;
	img::make_view(view1, gr_caddy.width, gr_caddy.height, buffer);
	img::map(gr_caddy, view1);

	img::View1r32 binary;
	img::make_view(binary, width / 2, height / 2, buffer);
	img::fill(binary, -1.0f);

	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = binary.width;
	r.y_begin = 0;
	r.y_end = 5;
	img::fill(img::sub_view(binary, r), 1.0f);
	r.y_begin = binary.height - 5;
	r.y_end = binary.height;
	img::fill(img::sub_view(binary, r), 1.0f);
	r.y_begin = 0;
	r.x_end = 5;
	img::fill(img::sub_view(binary, r), 1.0f);
	r.x_begin = binary.width - 5;
	r.x_end = binary.width;
	img::fill(img::sub_view(binary, r), 1.0f);

	r.x_begin = width / 4;
	r.x_end = r.x_begin + binary.width;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + binary.height;

	auto sub3 = img::sub_view(view3, r);
	img::overlay(sub3, binary, img::to_pixel(0, 255, 0), sub3);
	img::map_rgb(view3, vette);
	write_image(vette, "overlay.bmp");

	auto sub1 = img::sub_view(view1, r);
	img::overlay(sub1, binary, 255, sub1);
	img::map(view1, gr_caddy);
	write_image(gr_caddy, "overlay_gray.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_caddy);
	buffer.free();
}


void scale_down_test()
{
	auto title = "scale_down_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	img::read_image_from_file(CHESS_PATH, image);
	auto width = image.width;
	auto height = image.height;

	img::Buffer32 buffer(width * height * 5);

	img::View3r32 src3;
	img::make_view(src3, width, height, buffer);
	img::map_rgb(image, src3);

	auto scaled3 = img::scale_down(src3, buffer);

	Range2Du32 r{};
	r.x_begin = src3.width / 4;
	r.x_end = r.x_begin + scaled3.width;
	r.y_begin = src3.height / 4;
	r.y_end = r.y_begin + scaled3.height;

	auto view3 = img::sub_view(src3, r);
	img::copy(scaled3, view3);

	img::map_rgb(src3, image);	
	write_image(image, "scale_down3.bmp");

	GrayImage gray;
	img::read_image_from_file(CHESS_PATH, gray);
	width = gray.width;
	height = gray.height;

	buffer.reset();

	img::View1r32 src1;
	img::make_view(src1, width, height, buffer);
	img::map(gray, src1);

	auto scaled1 = img::scale_down(src1, buffer);

	r.x_begin = src3.width / 4;
	r.x_end = r.x_begin + scaled3.width;
	r.y_begin = src3.height / 4;
	r.y_end = r.y_begin + scaled3.height;

	auto view1 = img::sub_view(src1, r);
	img::copy(scaled1, view1);

	img::map(src1, gray);
	write_image(gray, "scale_down1.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
	buffer.free();
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