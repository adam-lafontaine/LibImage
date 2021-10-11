#include "../libimage/libimage.hpp"
//#include "../libimage/proc/process.hpp"
#include "./utils/stopwatch.hpp"

#include <cstdio>
#include <iostream>

namespace fs = std::filesystem;
namespace img = libimage;

using Image = img::image_t;
using ImageView = img::view_t;
using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using Pixel = img::pixel_t;

//constexpr auto ROOT_DIR = "~/Repos/LibImage/CudaTests";
constexpr auto ROOT_DIR = "/home/adam/Repos/LibImage/CudaTests";

const auto ROOT_PATH = fs::path(ROOT_DIR);

// make sure these files exist
const auto CORVETTE_PATH = ROOT_PATH / "in_files/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH / "in_files/cadillac.png";

const auto DST_IMAGE_ROOT = ROOT_PATH / "out_files";

void empty_dir(fs::path const& dir);
void process_tests(fs::path const& out_dir);

int main()
{
	auto dst_root = fs::path(DST_IMAGE_ROOT);

    process_tests(dst_root);

    printf("\nDone.\n");
}


void process_tests(fs::path const& out_dir)
{
	std::cout << "process:\n";
	empty_dir(out_dir);

	// get image
	Image corvette_image;
	img::read_image_from_file(CORVETTE_PATH, corvette_image);
	auto const width = corvette_image.width;
	auto const height = corvette_image.height;
	auto corvette_view = img::make_view(corvette_image);

	Image dst_image;
	img::make_image(dst_image, width, height);

	GrayImage dst_gray_image;
	img::make_image(dst_gray_image, width, height);

	// get another image for blending
	// make sure it is the same size
	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	Image caddy;
	caddy.width = width;
	caddy.height = height;
	img::resize_image(caddy_read, caddy);
	auto caddy_view = make_view(caddy);

	/*

	// alpha blending
	img::transform_alpha(caddy_view, [](auto const& p) { return 128; });
	img::alpha_blend(caddy_view, corvette_view, dst_image);
	img::write_image(dst_image, out_dir / "alpha_blend.png");

	img::copy(corvette_view, dst_image);
	img::alpha_blend(caddy_view, dst_image);
	img::write_image(dst_image, out_dir / "alpha_blend_src_dst.png");

	// grayscale
	img::transform_grayscale(corvette_view, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "convert_grayscale.png");
	
	// stats
	auto gray_stats = img::calc_stats(dst_gray_image);
	GrayImage gray_stats_image;
	img::draw_histogram(gray_stats.hist, gray_stats_image);
	img::write_image(gray_stats_image, out_dir / "gray_stats.png");
	print(gray_stats);

	// alpha grayscale
	img::transform_alpha_grayscale(corvette_view);
	auto alpha_stats = img::calc_stats(corvette_image, img::Channel::Alpha);
	GrayImage alpha_stats_image;
	img::draw_histogram(alpha_stats.hist, alpha_stats_image);
	img::write_image(alpha_stats_image, out_dir / "alpha_stats.png");
	print(alpha_stats);

	// create a new grayscale source
	GrayImage src_gray_image;
	img::make_image(src_gray_image, width, height);
	img::copy(dst_gray_image, src_gray_image);

	// contrast
	auto shade_min = static_cast<u8>(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = static_cast<u8>(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));
	img::transform_contrast(src_gray_image, dst_gray_image, shade_min, shade_max);
	img::write_image(dst_gray_image, out_dir / "contrast.png");

	// binarize
	auto const is_white = [&](u8 p) { return static_cast<r32>(p) > gray_stats.mean; };
	img::binarize(src_gray_image, dst_gray_image, is_white);
	img::write_image(dst_gray_image, out_dir / "binarize.png");

	//blur
	img::blur(src_gray_image, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "blur.png");	

	// edge detection
	img::edges(src_gray_image, dst_gray_image, 150);
	img::write_image(dst_gray_image, out_dir / "edges.png");

	// gradient
	img::gradients(src_gray_image, dst_gray_image);
	img::write_image(dst_gray_image, out_dir / "gradient.png");

	// combine transformations in the same image
	// regular grayscale to start
	img::copy(src_gray_image, dst_gray_image);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_image, range);
	auto dst_sub = img::sub_view(dst_gray_image, range);
	img::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::transform_contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::blur(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_image, range);
	dst_sub = img::sub_view(dst_gray_image, range);
	img::gradients(src_sub, dst_sub);

	img::write_image(dst_gray_image, out_dir / "combo.png");


	// compare edge detection speeds
	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::data_color_t seq_times;
	seq_times.color = green;

	img::data_color_t par_times;
	par_times.color = blue;

	Stopwatch sw;
	u32 size_start = 10000;

	u32 size = size_start;
	auto const scale = [&](auto t) { return static_cast<r32>(10000 * t / size); };

	for (u32 i = 0; i < 10; ++i, size *= 2)
	{
		GrayImage image;
		make_image(image, size);
		GrayImage dst;
		make_image(dst, size);
		auto view = img::make_view(image);
		auto dst_view = img::make_view(dst);

		sw.start();
		img::seq::edges(view, dst_view, 150);
		auto t = sw.get_time_milli();
		seq_times.data.push_back(scale(t));

		sw.start();
		img::edges(view, dst_view, 150);
		t = sw.get_time_milli();
		par_times.data.push_back(scale(t));
	}

	Image view_chart;
	std::vector<img::data_color_t> view_data =
	{
		seq_times, par_times
	};

	img::draw_bar_chart(view_data, view_chart);
	img::write_image(view_chart, out_dir / "edges_times.png");

	*/
}


void empty_dir(fs::path const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}