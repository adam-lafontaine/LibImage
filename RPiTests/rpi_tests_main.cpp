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
constexpr auto ROOT_DIR = "/home/pi/Repos/LibImage/RPiTests/";

const auto ROOT_PATH = std::string(ROOT_DIR);

const auto CORVETTE_PATH = ROOT_PATH + "in_files/corvette.png";
const auto CADILLAC_PATH = ROOT_PATH + "in_files/cadillac.png";

const auto DST_IMAGE_ROOT = ROOT_PATH + "out_files/";


void empty_dir(path_t& dir);
void basic_tests(path_t& out_dir);
void process_tests(path_t& out_dir);
void gradient_times(path_t& out_dir);


int main()
{	
	auto dst_root = DST_IMAGE_ROOT;

	//auto dst_basic = dst_root + "basic/";
    //basic_tests(dst_basic);

	//auto dst_process = dst_root + "process/";
	//process_tests(dst_process);

	auto dst_gradients = dst_root + "gradients/";
	gradient_times(dst_gradients);
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

	image.dispose();
	image_gray.dispose();

	printf("\n");
}


void process_tests(path_t& out_dir)
{
	printf("\nprocess:\n");
	empty_dir(out_dir);

	Stopwatch sw;
	sw.start();

	// get image
	Image corvette_img;
	img::read_image_from_file(CORVETTE_PATH, corvette_img);
	img::write_image(corvette_img, out_dir + "vette.png");

	auto const width = corvette_img.width;
	auto const height = corvette_img.height;	

	// get another image for blending
	// make sure it is the same size
	Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);
	Image caddy_img;
	caddy_img.width = width;
	caddy_img.height = height;
	img::resize_image(caddy_read, caddy_img);
	img::write_image(caddy_img, out_dir + "caddy.png");

	Image dst_img;
	img::make_image(dst_img, width, height);

	GrayImage dst_gray_img;
	img::make_image(dst_gray_img, width, height);

	// alpha blending
	img::transform_alpha(caddy_img, [](auto const& p) { return 128; });

	img::alpha_blend(caddy_img, corvette_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend.png");

	img::seq::alpha_blend(caddy_img, corvette_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_seq.png");

	img::simd::alpha_blend(caddy_img, corvette_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_simd.png");

	img::copy(corvette_img, dst_img);

	img::alpha_blend(caddy_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst.png");

	img::seq::alpha_blend(caddy_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst_seq.png");

	img::simd::alpha_blend(caddy_img, dst_img);
	img::write_image(dst_img, out_dir + "alpha_blend_src_dst_simd.png");


	// grayscale
	img::grayscale(corvette_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "grayscale.png");

	img::seq::grayscale(corvette_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "grayscale_seq.png");

	img::simd::grayscale(corvette_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "grayscale_simd.png");

	// stats
	auto gray_stats = img::calc_stats(dst_gray_img);
	GrayImage gray_stats_img;
	img::draw_histogram(gray_stats.hist, gray_stats_img);
	img::write_image(gray_stats_img, out_dir + "gray_stats.png");


	// alpha grayscale
	GrayImage alpha_stats_img;
	img::alpha_grayscale(corvette_img);
	auto alpha_stats = img::calc_stats(corvette_img, img::Channel::Alpha);
	
	img::draw_histogram(alpha_stats.hist, alpha_stats_img);
	img::write_image(alpha_stats_img, out_dir + "gray_stats_alpha.png");

	GrayImage alpha_stats_seq_img;
	img::seq::alpha_grayscale(corvette_img);
	alpha_stats = img::calc_stats(corvette_img, img::Channel::Alpha);
	
	img::draw_histogram(alpha_stats.hist, alpha_stats_seq_img);
	img::write_image(alpha_stats_img, out_dir + "gray_stats_alpha_seq.png");

	GrayImage alpha_stats_simd_img;
	img::simd::alpha_grayscale(corvette_img);
	alpha_stats = img::calc_stats(corvette_img, img::Channel::Alpha);
	
	img::draw_histogram(alpha_stats.hist, alpha_stats_simd_img);
	img::write_image(alpha_stats_img, out_dir + "gray_stats_alpha_simd.png");


	// create a new grayscale source
	GrayImage src_gray_img;
	img::make_image(src_gray_img, width, height);
	img::copy(dst_gray_img, src_gray_img);


	// contrast
	auto shade_min = (u8)(std::max(0.0f, gray_stats.mean - gray_stats.std_dev));
	auto shade_max = (u8)(std::min(255.0f, gray_stats.mean + gray_stats.std_dev));

	img::contrast(src_gray_img, dst_gray_img, shade_min, shade_max);
	img::write_image(dst_gray_img, out_dir + "contrast.png");

	img::seq::contrast(src_gray_img, dst_gray_img, shade_min, shade_max);
	img::write_image(dst_gray_img, out_dir + "contrast_seq.png");


	// binarize
	auto const is_white = [&](u8 p) { return (r32)(p) > gray_stats.mean; };

	img::binarize(src_gray_img, dst_gray_img, is_white);
	img::write_image(dst_gray_img, out_dir + "binarize.png");

	img::seq::binarize(src_gray_img, dst_gray_img, is_white);
	img::write_image(dst_gray_img, out_dir + "binarize_seq.png");


	//blur
	img::blur(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur.png");

	img::seq::blur(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur_seq.png");

	img::simd::blur(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "blur_simd.png");


	// edge detection
	auto const threshold = [](u8 g) { return g >= 100; };

	img::edges(src_gray_img, dst_gray_img, threshold);
	img::write_image(dst_gray_img, out_dir + "edges.png");

	img::seq::edges(src_gray_img, dst_gray_img, threshold);
	img::write_image(dst_gray_img, out_dir + "edges_seq.png");

	img::simd::edges(src_gray_img, dst_gray_img, threshold);
	img::write_image(dst_gray_img, out_dir + "edges_simd.png");


	// gradient
	img::gradients(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "gradient.png");

	img::seq::gradients(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "gradient_seq.png");

	img::simd::gradients(src_gray_img, dst_gray_img);
	img::write_image(dst_gray_img, out_dir + "gradient_simd.png");


	// combine transformations in the same image
	// regular grayscale to start
	img::seq::copy(src_gray_img, dst_gray_img);

	img::pixel_range_t range;
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src_gray_img, range);
	auto dst_sub = img::sub_view(dst_gray_img, range);
	img::gradients(src_sub, dst_sub);
	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);	
	img::contrast(src_sub, dst_sub, shade_min, shade_max);

	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::blur(src_sub, dst_sub);	

	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::binarize(src_sub, dst_sub, is_white);	

	range.x_begin = width / 4;
	range.x_end = range.x_begin + width / 2;
	range.y_begin = height / 4;
	range.y_end = range.y_begin + height / 2;
	src_sub = img::sub_view(src_gray_img, range);
	dst_sub = img::sub_view(dst_gray_img, range);
	img::edges(src_sub, dst_sub, threshold);

	img::write_image(dst_gray_img, out_dir + "combo.png");

	corvette_img.dispose();
	caddy_read.dispose();
	caddy_img.dispose();
	dst_img.dispose();
	dst_gray_img.dispose();
	gray_stats_img.dispose();
	alpha_stats_img.dispose();
	alpha_stats_seq_img.dispose();
	alpha_stats_simd_img.dispose();
	src_gray_img.dispose();

	auto time = sw.get_time_sec();

	printf("Done %f seconds\n", time);
}


void gradient_times(path_t& out_dir)
{
	printf("\ngradients:\n");
	empty_dir(out_dir);


	u32 n_image_sizes = 2;
	u32 image_dim_factor = 4;

	u32 n_image_counts = 2;
	u32 image_count_factor = 4;

	u32 width_start = 400;
	u32 height_start = 300;
	u32 image_count_start = 50;

	auto green = img::to_pixel(88, 100, 29);
	auto blue = img::to_pixel(0, 119, 182);

	img::multi_chart_data_t seq_image_times;
	seq_image_times.color = green;

	img::multi_chart_data_t seq_view_times;
	seq_view_times.color = blue;

	img::multi_chart_data_t par_image_times;
	par_image_times.color = green;

	img::multi_chart_data_t par_view_times;
	par_view_times.color = blue;

	img::multi_chart_data_t simd_image_times;
	simd_image_times.color = green;

	img::multi_chart_data_t simd_view_times;
	simd_view_times.color = blue;

	Stopwatch sw;
	u32 width = width_start;
	u32 height = height_start;
	u32 image_count = image_count_start;

	auto const current_pixels = [&]() { return (u64)(width) * height * image_count; };

	auto const start_pixels = current_pixels();

	auto const scale = [&](auto t) { return (r32)(start_pixels) / current_pixels() * t; };
	auto const print_wh = [&]() { printf("\nwidth: %u height: %u\n", width, height); };
	auto const print_count = [&]() { printf("  image count: %u\n", image_count); };

	r64 t = 0;
	auto const print_t = [&](const char* label) { printf("    %s time: %f\n", label, scale(t)); };

	for (u32 s = 0; s < n_image_sizes; ++s)
	{
		print_wh();
		image_count = image_count_start;
		std::vector<r32> seq_image;
		std::vector<r32> seq_view;
		std::vector<r32> par_image;
		std::vector<r32> par_view;
		std::vector<r32> simd_image;
		std::vector<r32> simd_view;
		GrayImage src;
		GrayImage dst;
		GrayImage tmp;
		img::make_image(src, width, height);
		img::make_image(dst, width, height);
		img::make_image(tmp, width, height);
		auto src_v = img::make_view(src);
		auto dst_v = img::make_view(dst);

		for (u32 c = 0; c < n_image_counts; ++c)
		{
			print_count();

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::seq::gradients(src, dst, tmp);
			}			
			t = sw.get_time_milli();
			seq_image.push_back(scale(t));
			print_t(" seq image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::seq::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			seq_view.push_back(scale(t));
			print_t("  seq view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			par_image.push_back(scale(t));
			print_t(" par image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			par_view.push_back(scale(t));
			print_t("  par view");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::simd::gradients(src, dst, tmp);
			}
			t = sw.get_time_milli();
			simd_image.push_back(scale(t));
			print_t("simd image");

			sw.start();
			for (u32 i = 0; i < image_count; ++i)
			{
				img::simd::gradients(src_v, dst_v, tmp);
			}
			t = sw.get_time_milli();
			simd_view.push_back(scale(t));
			print_t(" simd view");

			image_count *= image_count_factor;
		}

		seq_image_times.data_list.push_back(seq_image);
		seq_view_times.data_list.push_back(seq_view);
		par_image_times.data_list.push_back(par_image);
		par_view_times.data_list.push_back(par_view);
		simd_image_times.data_list.push_back(simd_image);
		simd_view_times.data_list.push_back(simd_view);

		width *= image_dim_factor;
		height *= image_dim_factor;
	}

	img::grouped_multi_chart_data_t chart_data
	{ 
		seq_image_times, seq_view_times,
		par_image_times, par_view_times,
		simd_image_times, simd_view_times
	};
	Image chart;
	img::draw_bar_multi_chart_grouped(chart_data, chart);
	img::write_image(chart, out_dir + "gradients.bmp");



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