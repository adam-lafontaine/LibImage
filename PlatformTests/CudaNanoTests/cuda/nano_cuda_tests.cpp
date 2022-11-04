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

void read_write_image_test();
void resize_test();
void map_test();
void map_rgb_test();
void sub_view_test();


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

	img::Buffer32 buffer(width * height, cuda::Malloc::Device);

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
	auto out_dir = IMAGE_OUT_PATH + title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir + name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	Image image_dst;
	img::make_image(image_dst, width, height);

	img::Buffer32 buffer(width * height * 4, cuda::Malloc::Device);

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
	
	img::Buffer32 buffer(width * height * 7, cuda::Malloc::Device);

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