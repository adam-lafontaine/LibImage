#include "../libimage/libimage.hpp"
#include "../libimage/proc/process.hpp"
#include "../libimage/math/math.hpp"
#include "../libimage/math/charts.hpp"
#include "./utils/stopwatch.hpp"

#include <cstdio>
#include <string>

namespace img = libimage;
using path_t = std::string;

using Image = img::image_t;
using ImageView = img::view_t;
using Pixel = img::pixel_t;

using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using GrayPixel = img::gray::pixel_t;

// make sure these files exist
constexpr auto ROOT_DIR = "/home/adam/Repos/LibImage/CudaTests/";

const auto ROOT_PATH = std::string(ROOT_DIR);

const auto CORVETTE_PATH = ROOT_PATH + "in_files/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH + "in_files/cadillac.png";

const auto DST_IMAGE_ROOT = ROOT_PATH + "out_files/";


void empty_dir(path_t& dir);
void basic_tests(path_t& out_dir);
void process_tests(path_t& out_dir);


int main()
{	
	auto dst_root = DST_IMAGE_ROOT;

	auto dst_basic = dst_root + "basic/";
    basic_tests(dst_basic);
}


void basic_tests(path_t& out_dir)
{
	printf("\nbasic:\n");
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);

	// write different file types
	img::write_image(image, out_dir + "image.png");
	img::write_image(image, out_dir + "image.bmp");

	// write views with different file types
	auto view = img::make_view(image);
	img::write_view(view, out_dir + "view.png");
	img::write_view(view, out_dir + "view.bmp");

	auto w = view.width;
	auto h = view.height;

	// get a portion of an existing image
	img::pixel_range_t range = { w * 1 / 3, w * 2 / 3, h * 1 / 3, h * 2 / 3 };
	auto sub_view = img::sub_view(view, range);
	img::write_view(sub_view, out_dir + "sub.png");

	// get one row from an image
	auto row_view = img::row_view(view, h / 2);
	img::write_view(row_view, out_dir + "row_view.bmp");

	// get one column from an image
	auto col_view = img::column_view(view, w / 2);
	img::write_view(col_view, out_dir + "col_view.bmp");

	// resize an image
	Image resize_image;
	resize_image.width = w / 4;
	resize_image.height = h / 2;
	auto resize_view = img::make_resized_view(image, resize_image);
	img::write_image(resize_image, out_dir + "resize_image.bmp");
	img::write_view(resize_view, out_dir + "resize_view.bmp");

	// read a color image to grayscale
	GrayImage image_gray;
	img::read_image_from_file(CADILLAC_PATH, image_gray);
	img::write_image(image_gray, out_dir + "image_gray.bmp");

	// create a grayscale view
	auto view_gray = img::make_view(image_gray);
	img::write_view(view_gray, out_dir + "view_gray.bmp");

	// portion of a grayscale image
	auto sub_view_gray = img::sub_view(view_gray, range);
	img::write_view(sub_view_gray, out_dir + "sub_view_gray.png");

	// row from a grayscale image
	auto row_view_gray = img::row_view(view_gray, view_gray.height / 2);
	img::write_view(row_view_gray, out_dir + "row_view_gray.png");

	// column from a grayscale image
	auto col_view_gray = img::column_view(view_gray, view_gray.width / 2);
	img::write_view(col_view_gray, out_dir + "col_view_gray.png");

	// resize a grayscale image
	GrayImage resize_image_gray;
	resize_image_gray.width = w / 4;
	resize_image_gray.height = h / 2;
	auto resize_view_gray = img::make_resized_view(image_gray, resize_image_gray);
	img::write_image(resize_image_gray, out_dir + "resize_image_gray.bmp");
	img::write_view(resize_view_gray, out_dir + "resize_view_gray.bmp");

	printf("\n");
}


void process_tests(path_t& out_dir)
{
	printf("\nprocess:\n");
	empty_dir(out_dir);

	// get image
	Image corvette_img;
	img::read_image_from_file(CORVETTE_PATH, corvette_img);
	img::write_image(corvette_img, out_dir + "vette.png");


	corvette_img.dispose();
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

	command = std::string("rm -rfv ") + dir + '*';
	system(command.c_str());	
}