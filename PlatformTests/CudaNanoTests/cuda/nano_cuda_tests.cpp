#include "../utils/stopwatch.hpp"
#include "../../../libimage_cuda/libimage.hpp"

#include <cstdio>
#include <locale.h>
#include <string>

namespace img = libimage;

using Image = img::Image;
using ImageView = img::View;
using GrayImage = img::gray::Image;
using GrayView = img::gray::View;
using Pixel = img::Pixel;

using path_t = std::string;

namespace fs
{
	bool is_directory(path_t const&);

	bool exists(path_t const&);
}


// set this directory for your system
constexpr auto ROOT_DIR = "../../../";

constexpr auto TEST_IMAGE_DIR = "TestImages/";
constexpr auto IMAGE_IN_DIR = "in_files/";
constexpr auto IMAGE_OUT_DIR = "out_files/";

const auto ROOT_PATH = path_t(ROOT_DIR);
const auto TEST_IMAGE_PATH = ROOT_PATH + TEST_IMAGE_DIR;
const auto IMAGE_IN_PATH = TEST_IMAGE_PATH + IMAGE_IN_DIR;
const auto IMAGE_OUT_PATH = TEST_IMAGE_PATH + IMAGE_OUT_DIR;

const auto CORVETTE_PATH = IMAGE_IN_PATH + "corvette.png";
const auto CADILLAC_PATH = IMAGE_IN_PATH + "cadillac.png";
const auto WEED_PATH = IMAGE_IN_PATH + "weed.png";
const auto CHESS_PATH = IMAGE_IN_PATH + "chess_board.bmp";


bool directory_files_test()
{
	auto title = "directory_files_test";
	printf("\n%s:\n", title);

	auto const test_dir = [](path_t const& dir)
	{
		auto result = fs::is_directory(dir);
		auto msg = result ? "PASS" : "FAIL";
		printf("%s: %s\n", dir.c_str(), msg);

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
		printf("%s: %s\n", file.c_str(), msg);

		return result;
	};

	result =
		result &&
		test_file(CORVETTE_PATH) &&
		test_file(CADILLAC_PATH) &&
		test_file(WEED_PATH);

	return result;
}


void empty_dir(path_t& dir);
void clear_image(Image const& img);
void clear_image(GrayImage const& img);

void read_write_image_test();
void resize_test();
void map_test();
void map_rgb_test();
void sub_view_test();
void select_channel_test();
void map_hsv_test();
void fill_test();
void copy_test();
void grayscale_test();
void alpha_blend_test();
void threshold_test();
void contrast_test();
void blur_test();


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
	sub_view_test();
	select_channel_test();
	map_hsv_test();
	fill_test();
	copy_test();
	grayscale_test();
	alpha_blend_test();
	threshold_test();
	contrast_test();
	blur_test();


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
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

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
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

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
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	auto width = gray.width;
	auto height = gray.height;

	GrayImage gray_dst;
	img::make_image(gray_dst, width, height);

	img::DeviceBuffer32 d_buffer(width * height, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height);

	img::View1r32 view1;
	img::make_view(view1, width, height, d_buffer);

	img::map(img::make_view(gray), view1, h_buffer);
	img::map(view1, img::make_view(gray_dst), h_buffer);

	write_image(gray_dst, "map1.bmp");
	
	img::destroy_image(gray);
	img::destroy_image(gray_dst);
	d_buffer.free();
	h_buffer.free();
}


void map_rgb_test()
{
	auto title = "map_rgb_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image image_dst;
	img::make_image(image_dst, width, height);

	auto view = img::make_view(image);
	auto view_dst = img::make_view(image_dst);

	img::DeviceBuffer32 d_buffer(width * height * 4, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View4r32 view4;
	img::make_view(view4, width, height, d_buffer);

	img::map_rgb(view, view4, h_buffer);
	img::map_rgb(view4, view_dst, h_buffer);
	d_buffer.reset();

	write_image(image_dst, "map_rgba.bmp");

	img::View3r32 view3;
	img::make_view(view3, width, height, d_buffer);

	img::map_rgb(view, view3, h_buffer);
	img::map_rgb(view3, view_dst, h_buffer);
	d_buffer.reset();

	write_image(image_dst, "map_rgb.bmp");

	img::destroy_image(image);
	img::destroy_image(image_dst);
	d_buffer.free();
	h_buffer.free();
}


void sub_view_test()
{
	auto title = "sub_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	auto view = img::make_view(vette);
	
	img::DeviceBuffer32 d_buffer(width * height * 7, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View3r32 vette3;
	img::make_view(vette3, width, height, d_buffer);
	img::map_rgb(view, vette3, h_buffer);

	img::View4r32 vette4;
	img::make_view(vette4, width, height, d_buffer);
	img::map_rgb(view, vette4, h_buffer);

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

	img::map_rgb(sub3, dst3, h_buffer);
	img::map_rgb(sub4, dst4, h_buffer);
	d_buffer.reset();

	write_image(vette, "swap.bmp");

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	width = caddy.width;
	height = caddy.height;

	img::View1r32 caddy1;
	img::make_view(caddy1, width, height, d_buffer);
	img::map(img::make_view(caddy), caddy1, h_buffer);

	r.x_begin = 0;
	r.x_end = width / 2;
	r.y_begin = 0;
	r.y_end = height;

	auto sub1 = img::sub_view(caddy1, r);

	r.x_begin = width / 4;
	r.x_end = width * 3 / 4;

	auto dst1 = img::sub_view(caddy, r);

	img::map(sub1, dst1, h_buffer);

	write_image(caddy, "copy.bmp");		

	img::destroy_image(vette);
	img::destroy_image(caddy);
	d_buffer.free();
	h_buffer.free();
}


void select_channel_test()
{
	auto title = "select_channel_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image vette_img;
	img::read_image_from_file(CORVETTE_PATH, vette_img);
	auto width = vette_img.width;
	auto height = vette_img.height;

	auto vette = img::make_view(vette_img);
	
	img::DeviceBuffer32 d_buffer(width * height * 7, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View3r32 vette3;
	img::make_view(vette3, width, height, d_buffer);
	img::map_rgb(vette, vette3, h_buffer);

	GrayImage vette_dst;
	img::make_image(vette_dst, vette.width, vette.height);

	Image caddy_img;
	img::read_image_from_file(CADILLAC_PATH, caddy_img);
	auto caddy = img::make_view(caddy_img);

	img::View4r32 caddy4;
	img::make_view(caddy4, caddy.width, caddy.height, d_buffer);
	img::map_rgb(caddy, caddy4, h_buffer);

	GrayImage caddy_dst;
	img::make_image(caddy_dst, caddy.width, caddy.height);

	auto red = img::select_channel(vette3, img::RGB::R);
	auto blue = img::select_channel(caddy4, img::RGBA::B);

	img::map(red, img::make_view(vette_dst), h_buffer);
	write_image(vette_dst, "red.bmp");

	img::map(blue, img::make_view(caddy_dst), h_buffer);
	write_image(caddy_dst, "blue.bmp");

	Image image_dst;
	img::make_image(image_dst, width, height);
	auto view_dst = img::make_view(image_dst);

	img::map_rgb(vette, vette3, h_buffer);
	auto view_ch = img::select_channel(vette3, img::RGB::R);
	img::multiply(view_ch, 0.5f);
	img::map_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "reduce_r.bmp");

	img::map_rgb(vette, vette3, h_buffer);
	view_ch = img::select_channel(vette3, img::RGB::G);
	img::multiply(view_ch, 0.5f);
	img::map_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "reduce_g.bmp");

	img::map_rgb(vette, vette3, h_buffer);
	view_ch = img::select_channel(vette3, img::RGB::B);
	img::multiply(view_ch, 0.5f);
	img::map_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "reduce_b.bmp");

	img::map_rgb_hsv(vette, vette3, h_buffer);
	view_ch = img::select_channel(vette3, img::HSV::H);
	img::fill(view_ch, 0.5f);
	img::map_hsv_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "change_h.bmp");

	img::map_rgb_hsv(vette, vette3, h_buffer);
	view_ch = img::select_channel(vette3, img::HSV::S);
	img::multiply(view_ch, 0.5f);
	img::map_hsv_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "reduce_s.bmp");

	img::map_rgb_hsv(vette, vette3, h_buffer);
	view_ch = img::select_channel(vette3, img::HSV::V);
	img::multiply(view_ch, 0.5f);
	img::map_hsv_rgb(vette3, view_dst, h_buffer);
	write_image(image_dst, "reduce_v.bmp");

	img::destroy_image(vette_img);
	img::destroy_image(vette_dst);
	img::destroy_image(caddy_img);
	img::destroy_image(caddy_dst);
	img::destroy_image(image_dst);
	d_buffer.free();
	h_buffer.free();
}


void map_hsv_test()
{
	auto title = "map_hsv_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image image_dst;
	img::make_image(image_dst, width, height);

	img::DeviceBuffer32 d_buffer(3 * width * height * 3, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(3 * width * height);

	img::ViewRGBr32 rgb_src;
	img::make_view(rgb_src, width, height, d_buffer);

	img::ViewHSVr32 hsv;
	img::make_view(hsv, width, height, d_buffer);

	img::ViewRGBr32 rgb_dst;
	img::make_view(rgb_dst, width, height, d_buffer);

	img::map_rgb(img::make_view(image), rgb_src, h_buffer);

	img::map_rgb_hsv(rgb_src, hsv);

	img::map_hsv_rgb(hsv, rgb_dst);

	img::map_rgb(rgb_dst, img::make_view(image_dst), h_buffer);

	write_image(image_dst, "map_hsv.bmp");

	img::destroy_image(image);
	img::destroy_image(image_dst);
	d_buffer.free();
	h_buffer.free();
}


void fill_test()
{
	auto title = "fill_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

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

	auto view = img::make_view(image);
	auto left_view = img::sub_view(image, left);
	auto top_left_view = img::sub_view(image, top_left);
	auto bottom_right_view = img::sub_view(image, bottom_right);

	img::fill(view, red);
	write_image(image, "fill_01.bmp");

	img::fill(left_view, green);
	write_image(image, "fill_02.bmp");

	GrayImage gray;
	img::make_image(gray, width, height);

	auto gr_top_left_view = img::sub_view(gray, top_left);
	auto gr_bottom_right_view = img::sub_view(gray, bottom_right);

	img::fill(img::make_view(gray), 128);
	write_image(gray, "gray_fill_01.bmp");

	img::fill(gr_top_left_view, 0);
	img::fill(gr_bottom_right_view, 255);
	write_image(gray, "gray_fill_02.bmp");
	
	img::DeviceBuffer32 d_buffer(width * height * 4, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);
	
	img::View3r32 view3;
	img::make_view(view3, width / 2, height / 2, d_buffer);
	img::fill(view3, blue);
	img::map_rgb(view3, top_left_view, h_buffer);
	d_buffer.reset();
	write_image(image, "fill_03.bmp");

	img::View4r32 view4;
	img::make_view(view4, width / 2, height / 2, d_buffer);

	img::fill(view4, white);
	img::map_rgb(view4, bottom_right_view, h_buffer);
	d_buffer.reset();
	write_image(image, "fill_04.bmp");


	img::make_view(view3, width, height, d_buffer);
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
	img::map_rgb(view3, view, h_buffer);
	d_buffer.reset();
	write_image(image, "fill_view3.bmp");

	img::make_view(view4, width, height, d_buffer);
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
	img::map_rgb(view4, view, h_buffer);
	d_buffer.reset();
	write_image(image, "fill_view4.bmp");	

	img::destroy_image(image);
	img::destroy_image(gray);
	d_buffer.free();
	h_buffer.free();
}


void copy_test()
{
	auto title = "copy_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	auto view = img::make_view(image);

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
	
	img::DeviceBuffer32 d_buffer(width * height * 7, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View3r32 view3;
	img::make_view(view3, width, height, d_buffer);
	img::map_rgb(view, view3, h_buffer);
	auto left_view3 = img::sub_view(view3, left);
	auto right_view3 = img::sub_view(view3, right);

	img::View4r32 view4;
	img::make_view(view4, width, height, d_buffer);
	img::map_rgb(view, view4, h_buffer);
	auto left_view4 = img::sub_view(view4, left);
	auto right_view4 = img::sub_view(view4, right);

	img::copy(left_view3, right_view3);
	img::copy(right_view4, left_view4);	

	clear_image(image);

	img::map_rgb(right_view3, right_view, h_buffer);
	img::map_rgb(left_view4, left_view, h_buffer);
	write_image(image, "image.bmp");

	d_buffer.reset();

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
	img::make_view(top1, width, view_height, d_buffer);
	img::map(gr_top_view, top1, h_buffer);

	img::View1r32 bottom1;
	img::make_view(bottom1, width, view_height, d_buffer);
	img::map(gr_bottom_view, bottom1, h_buffer);

	img::copy(bottom1, top1);

	img::map(top1, gr_top_view, h_buffer);

	write_image(gray, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
	d_buffer.free();
	h_buffer.free();
}


void grayscale_test()
{
	auto title = "grayscale_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

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
	
	img::DeviceBuffer32 d_buffer(width * height * 5, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View3r32 view3;
	img::make_view(view3, width / 2, height, d_buffer);

	img::View4r32 view4;
	img::make_view(view4, width / 2, height, d_buffer);

	img::map_rgb(left_view, view3, h_buffer);
	img::map_rgb(right_view, view4, h_buffer);

	GrayImage dst;
	img::make_image(dst, width, height);

	img::View1r32 view1;
	img::make_view(view1, width, height, d_buffer);

	auto gr_left = img::sub_view(view1, left);
	auto gr_right = img::sub_view(view1, right);

	img::grayscale(view3, gr_right);
	img::grayscale(img::make_rgb_view(view4), gr_left);

	img::map(view1, img::make_view(dst), h_buffer);

	write_image(image, "image.bmp");
	write_image(dst, "gray.bmp");

	img::destroy_image(image);
	img::destroy_image(dst);
	d_buffer.free();
	h_buffer.free();
}


void alpha_blend_test()
{
	auto title = "alpha_blend_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image vette_img;
	img::read_image_from_file(CORVETTE_PATH, vette_img);	
	
	auto width = vette_img.width;
	auto height = vette_img.height;

	GrayImage gr_vette_img;
	img::read_image_from_file(CORVETTE_PATH, gr_vette_img);	

	Image caddy_read_img;
	img::read_image_from_file(CADILLAC_PATH, caddy_read_img);

	GrayImage gr_caddy_read_img;
	img::read_image_from_file(CADILLAC_PATH, gr_caddy_read_img);	

	Image caddy_img;
	caddy_img.width = width;
	caddy_img.height = height;
	img::resize_image(caddy_read_img, caddy_img);

	GrayImage gr_caddy_img;
	gr_caddy_img.width = width;
	gr_caddy_img.height = height;
	img::resize_image(gr_caddy_read_img, gr_caddy_img);

	auto vette = img::make_view(vette_img);
	auto gr_vette = img::make_view(gr_vette_img);
	auto caddy = img::make_view(caddy_img);
	auto gr_caddy = img::make_view(gr_caddy_img);
	
	img::DeviceBuffer32 d_buffer(width * height * 11, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View4r32 vette4;
	img::make_view(vette4, width, height, d_buffer);
	img::map_rgb(vette, vette4, h_buffer);

	auto alpha_view = img::select_channel(vette4, img::RGBA::A);
	img::fill(alpha_view, 0.5f);

	img::View4r32 caddy4;
	img::make_view(caddy4, width, height, d_buffer);
	img::map_rgb(caddy, caddy4, h_buffer);

	img::View3r32 dst3;
	img::make_view(dst3, width, height, d_buffer);

	img::alpha_blend(vette4, img::make_rgb_view(caddy4), dst3);

	clear_image(vette_img);
	img::map_rgb(dst3, vette, h_buffer);
	write_image(vette_img, "blend_01.bmp");

	img::alpha_blend(vette4, img::make_rgb_view(caddy4), img::make_rgb_view(caddy4));

	clear_image(vette_img);
	img::map_rgb(caddy4, vette, h_buffer);
	write_image(vette_img, "blend_02.bmp");

	d_buffer.reset();

	img::View2r32 caddy2;
	img::make_view(caddy2, width, height, d_buffer);
	img::map(gr_caddy, img::select_channel(caddy2, img::GA::G), h_buffer);
	alpha_view = img::select_channel(caddy2, img::GA::A);
	img::fill(alpha_view, 0.5f);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, d_buffer);
	img::map(gr_vette, vette1, h_buffer);

	img::View1r32 dst1;
	img::make_view(dst1, width, height, d_buffer);

	img::alpha_blend(caddy2, vette1, dst1);

	img::map(dst1, gr_vette, h_buffer);
	write_image(gr_vette_img, "gr_blend.bmp");

	img::destroy_image(vette_img);
	img::destroy_image(gr_vette_img);
	img::destroy_image(caddy_read_img);
	img::destroy_image(gr_caddy_read_img);
	img::destroy_image(caddy_img);	
	img::destroy_image(gr_caddy_img);
	d_buffer.free();
}


void threshold_test()
{
	auto title = "threshold_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_dst;
	img::make_image(gr_dst, width, height);

	img::threshold(img::make_view(vette), img::make_view(gr_dst), 127, 255);

	write_image(gr_dst, "threshold.bmp");
	
	img::DeviceBuffer32 d_buffer(width * height, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, d_buffer);
	img::map(img::make_view(vette), vette1, h_buffer);

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	auto vette_left = img::sub_view(vette1, left);

	img::threshold(vette_left, vette_left, 0.0f, 0.5f);
	img::map(vette1, img::make_view(gr_dst), h_buffer);
	write_image(gr_dst, "threshold1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_dst);
	d_buffer.free();
	h_buffer.free();
}


void contrast_test()
{
	auto title = "contrast_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	GrayImage vette;
	img::read_image_from_file(CORVETTE_PATH, vette);
	auto width = vette.width;
	auto height = vette.height;

	GrayImage gr_dst;
	img::make_image(gr_dst, width, height);

	img::contrast(img::make_view(vette), img::make_view(gr_dst), 10, 175);

	write_image(gr_dst, "contrast.bmp");
	
	img::DeviceBuffer32 d_buffer(width * height, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height);

	img::View1r32 vette1;
	img::make_view(vette1, width, height, d_buffer);
	img::map(img::make_view(vette), vette1, h_buffer);

	Range2Du32 left{};
	left.x_begin = 0;
	left.x_end = width / 2;
	left.y_begin = 0;
	left.y_end = height;

	auto vette_left = img::sub_view(vette1, left);

	img::contrast(vette_left, vette_left, 0.1f, 0.75f);
	img::map(vette1, img::make_view(gr_dst), h_buffer);
	write_image(gr_dst, "contrast1.bmp");

	img::destroy_image(vette);
	img::destroy_image(gr_dst);
	d_buffer.free();
	h_buffer.free();
}


void blur_test()
{
	auto title = "blur_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	GrayImage vette_img;
	img::read_image_from_file(CORVETTE_PATH, vette_img);
	auto vette = img::make_view(vette_img);
	auto width = vette.width;
	auto height = vette.height;

	img::DeviceBuffer32 d_buffer(width * height * 6, cuda::Malloc::Device);
	img::HostBuffer32 h_buffer(width * height * 4);

	img::View1r32 src;
	img::make_view(src, width, height, d_buffer);

	img::View1r32 dst;
	img::make_view(dst, width, height, d_buffer);

	img::map(vette, src, h_buffer);

	img::blur(src, dst);

	img::map(dst, vette, h_buffer);

	write_image(vette_img, "blur1.bmp");

	d_buffer.reset();

	Image caddy_img;
	img::read_image_from_file(CADILLAC_PATH, caddy_img);
	auto caddy = img::make_view(caddy_img);
	width = caddy.width;
	height = caddy.height;

	img::View3r32 src3;
	img::make_view(src3, width, height, d_buffer);

	img::map_rgb(caddy, src3, h_buffer);

	img::View3r32 dst3;
	img::make_view(dst3, width, height, d_buffer);

	img::blur(src3, dst3);

	img::map_rgb(dst3, caddy, h_buffer);
	write_image(caddy_img, "blur3.bmp");

	img::destroy_image(vette_img);
	img::destroy_image(caddy_img);
	d_buffer.free();
	h_buffer.free();
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

	command = std::string("rm -rfv ") + dir + '*' + " > /dev/null";
	system(command.c_str());
}


void clear_image(Image const& img)
{
	constexpr auto black = img::to_pixel(0, 0, 0);
	img::fill(img::make_view(img), black);
}


void clear_image(GrayImage const& img)
{
	img::fill(img::make_view(img), 0);
}


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs
{
	bool is_directory(path_t const& dir)
	{
		struct stat st;
		if(stat(dir.c_str(), &st) == 0)
		{
			return (st.st_mode & S_IFDIR) != 0;
		}
		
		return false;
	}


	bool exists(path_t const& path)
	{
		// directory
		struct stat st;
		return stat(path.c_str(), &st) == 0;
	}
}