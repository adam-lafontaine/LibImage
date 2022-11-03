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


int main()
{
    Stopwatch sw;
	sw.start();

	if (!directory_files_test())
	{
		return EXIT_FAILURE;
	}

	read_write_image_test();


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