#include "../../../libimage_parallel/libimage.hpp"

#include "../utils/stopwatch.hpp"

#include <cstdio>
#include <algorithm>

namespace img = libimage;

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
void sub_view_test();
void transform_test();
void copy_test();
void fill_test();
void alpha_blend_test();
void grayscale_test();
void binary_test();
void contrast_test();
void blur_test();
void gradients_test();
void edges_test();
void combo_view_test();
void rotate_test();


int main()
{
	if (!directory_files_test())
	{
		return EXIT_FAILURE;
	}

	read_write_image_test();
	resize_test();
	sub_view_test();
	transform_test();
	copy_test();
	fill_test();
	alpha_blend_test();
	grayscale_test();
	binary_test();
	contrast_test();
	blur_test();
	gradients_test();
	edges_test();
	combo_view_test();
	rotate_test();
}


void read_write_image_test()
{
	auto title = "read_write_image_test";
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


void sub_view_test()
{
	auto title = "sub_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Range2Du32 r{};

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view = img::sub_view(image, r);

	Image dst;
	img::make_image(dst, view.width, view.height);
	img::copy(view, dst);

	write_image(dst, "sub_view.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view_gray = img::sub_view(gray, r);

	GrayImage dst_gray;
	img::make_image(dst_gray, view_gray.width, view_gray.height);
	img::copy(view_gray, dst_gray);

	write_image(dst_gray, "sub_view_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(gray);
	img::destroy_image(dst);
	img::destroy_image(dst_gray);
}


void transform_test()
{
	auto title = "transform_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	write_image(image, "vette.bmp");

	Image dst;
	img::make_image(dst, image.width, image.height);

	auto const invert = [](u8 p) { return 255 - p; };

	auto const invert_rgb = [&](Pixel p)
	{
		auto r = invert(p.red);
		auto g = invert(p.green);
		auto b = invert(p.blue);
		return img::to_pixel(r, g, b);
	};

	img::transform(img::make_view(image), img::make_view(dst), invert_rgb);
	write_image(dst, "transform.bmp");

	clear_image(dst);

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	write_image(gray, "caddy.bmp");

	GrayImage dst_gray;
	img::make_image(dst_gray, gray.width, gray.height);

	img::transform(img::make_view(gray), img::make_view(dst_gray), invert);
	write_image(dst_gray, "transform_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(dst);
	img::destroy_image(gray);
	img::destroy_image(dst_gray);
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
	write_image(image, "vette.bmp");

	Image dst;
	img::make_image(dst, image.width, image.height);

	img::copy(image, dst);
	write_image(dst, "copy.bmp");


	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	write_image(gray, "caddy.bmp");

	GrayImage dst_gray;
	img::make_image(dst_gray, gray.width, gray.height);

	img::copy(gray, dst_gray);
	write_image(dst_gray, "copy_gray.bmp");

	img::destroy_image(image);
	img::destroy_image(dst);
	img::destroy_image(gray);
	img::destroy_image(dst_gray);
}


void fill_test()
{
	auto title = "fill_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	u32 width = 800;
	u32 height = 600;

	Image dst;
	img::make_image(dst, width, height);

	img::fill(dst, img::to_pixel(20, 20, 220));
	write_image(dst, "fill.bmp");

	GrayImage dst_gray;
	img::make_image(dst_gray, width, height);

	img::fill(dst_gray, 127);
	write_image(dst_gray, "copy_gray.bmp");

	img::destroy_image(dst);
	img::destroy_image(dst_gray);
}


void alpha_blend_test()
{
	auto title = "alpha_blend_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image src_image;
	img::read_image_from_file(CORVETTE_PATH, src_image);
	auto width = src_image.width;
	auto height = src_image.height;	

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	Image cur_image;
	cur_image.width = width;
	cur_image.height = height;
	img::resize_image(caddy, cur_image);	

	Image dst_image;
	img::make_image(dst_image, width, height);

	auto src = img::make_view(src_image);
	auto cur = img::make_view(cur_image);
	auto dst = img::make_view(dst_image);

	img::transform_alpha(src, [](auto const& p) { return 128; });

	img::alpha_blend(src, cur, dst);
	write_image(dst_image, "alpha_blend.bmp");

	clear_image(dst_image);

	img::copy(cur, dst);
	img::alpha_blend(src, dst);
	write_image(dst_image, "alpha_blend_src_dst.bmp");

	img::destroy_image(src_image);
	img::destroy_image(caddy);
	img::destroy_image(cur_image);
	img::destroy_image(dst_image);
}


void grayscale_test()
{
	auto title = "grayscale_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	GrayImage dst;
	img::make_image(dst, width, height);

	img::grayscale(img::make_view(src), img::make_view(dst));
	write_image(dst, "grayscale.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void binary_test()
{
	auto title = "binary_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	write_image(caddy, "caddy.bmp");

	GrayImage caddy_binary;
	img::make_image(caddy_binary, caddy.width, caddy.height);

	auto src = img::make_view(caddy);
	auto dst = img::make_view(caddy_binary);

	img::binarize_th(src, dst, 128);
	write_image(caddy_binary, "caddy_binary.bmp");

	Image weed;
	img::read_image_from_file(WEED_PATH, weed);
	auto width = weed.width;
	auto height = weed.height;

	GrayImage binary_src;
	img::make_image(binary_src, width, height);

	GrayImage binary_dst;
	img::make_image(binary_dst, width, height);

	src = img::make_view(binary_src);
	dst = img::make_view(binary_dst);

	auto const is_white = [](Pixel p)
	{
		return ((r32)p.red + (r32)p.blue + (r32)p.green) / 3.0f < 190;
	};

	img::binarize(img::make_view(weed), src, is_white);
	write_image(binary_src, "weed.bmp");

	// centroid point
	auto pt = img::centroid(src);

	// region around centroid
	Range2Du32 c{};
	c.x_begin = pt.x - 10;
	c.x_end = pt.x + 10;
	c.y_begin = pt.y - 10;
	c.y_end = pt.y + 10;

	// draw binary image with centroid region
	img::copy(binary_src, binary_dst);
	auto c_view = img::sub_view(binary_dst, c);
	img::fill(c_view, 0);
	write_image(binary_dst, "centroid.bmp");

	// thin the object
	img::skeleton(src, dst);
	write_image(binary_dst, "skeleton.bmp");

	img::destroy_image(caddy);
	img::destroy_image(caddy_binary);
	img::destroy_image(weed);
	img::destroy_image(binary_src);
	img::destroy_image(binary_dst);
}


void contrast_test()
{
	auto title = "contrast_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	write_image(src, "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::contrast(img::make_view(src), img::make_view(dst), 0, 128);
	write_image(dst, "contrast.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void blur_test()
{
	auto title = "blur_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	write_image(src, "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::blur(img::make_view(src), img::make_view(dst));
	write_image(dst, "blur.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void gradients_test()
{
	auto title = "gradients_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	write_image(src, "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::gradients(img::make_view(src), img::make_view(dst));
	write_image(dst, "gradient.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void edges_test()
{
	auto title = "edges_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	write_image(src, "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	auto const threshold = [](u8 g) { return g >= 100; };

	img::edges(img::make_view(src), img::make_view(dst), threshold);
	write_image(dst, "edges.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void combo_view_test()
{
	auto title = "combo_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	GrayImage dst;
	img::make_image(dst, width, height);

	// top left
	Range2Du32 range{};
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
	auto src_sub = img::sub_view(src, range);
	auto dst_sub = img::sub_view(dst, range);

	img::gradients(src_sub, dst_sub);

	// top right
	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);

	auto const invert = [](u8 p) { return 255 - p; };
	img::transform(src_sub, dst_sub, invert);

	// bottom left
	range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = height / 2;
	range.y_end = height;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);

	img::binarize_th(src_sub, dst_sub, 75);

	// bottom right
	range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);

	auto const threshold = [](u8 g) { return g >= 100; };
	img::edges(src_sub, dst_sub, threshold);

	write_image(dst, "combo.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
}


void rotate_test()
{
	auto title = "rotate_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

	Image src;
	img::read_image_from_file(CADILLAC_PATH, src);

	Image dst;
	img::make_image(dst, src.width, src.height);

	Point2Du32 origin = { src.width / 2, src.height / 2 };

	r32 theta = 0.6f * 2 * 3.14159f;
	img::rotate(img::make_view(src), img::make_view(dst), origin, theta);
	write_image(dst, "rotate.bmp");

	GrayImage src_gray;
	img::read_image_from_file(CORVETTE_PATH, src_gray);

	GrayImage dst_gray;
	img::make_image(dst_gray, src_gray.width, src_gray.height);

	origin = { src_gray.width / 2, src_gray.height / 2 };

	img::rotate(img::make_view(src_gray), img::make_view(dst_gray), origin, theta);
	write_image(dst_gray, "rotate_gray.bmp");

	img::destroy_image(src);
	img::destroy_image(dst);
	img::destroy_image(src_gray);
	img::destroy_image(dst_gray);
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