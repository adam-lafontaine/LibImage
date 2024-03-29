#include "../../../libimage_compact/libimage.hpp"

#include "../utils/stopwatch.hpp"

//#define CHECK_LEAKS

#if defined(_WIN32) && defined(CHECK_LEAKS)
#include "../utils/win32_leak_check.h"
#endif

#include <cstdio>
#include <algorithm>

namespace img = libimage;

using Image = img::image_t;
using ImageView = img::view_t;
using GrayImage = img::gray::image_t;
using GrayView = img::gray::view_t;
using Pixel = img::pixel_t;

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
void view_test();
void transform_test();
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
	view_test();
	transform_test();
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

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	img::write_image(image, out_dir / "corvette.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	img::write_image(gray, out_dir / "cadillac_gray.bmp");
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
}


void view_test()
{
	auto title = "view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	img::pixel_range_t r{};

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto width = image.width;
	auto height = image.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view = img::sub_view(image, r);
	img::write_view(view, out_dir / "view.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	width = gray.width;
	height = gray.height;

	r.x_begin = width / 4;
	r.x_end = r.x_begin + width / 2;
	r.y_begin = height / 4;
	r.y_end = r.y_begin + height / 2;

	auto view_gray = img::sub_view(gray, r);
	img::write_view(view_gray, out_dir / "view_gray.bmp");
}


void transform_test()
{
	auto title = "transform_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image image;
	img::read_image_from_file(CORVETTE_PATH, image);
	img::write_image(image, out_dir / "vette.bmp");

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

	img::transform(image, dst, invert_rgb);
	img::write_image(dst, out_dir / "transform.bmp");

	clear_image(dst);

	img::seq::transform(image, dst, invert_rgb);
	img::write_image(dst, out_dir / "seq_transform.bmp");

	GrayImage gray;
	img::read_image_from_file(CADILLAC_PATH, gray);
	img::write_image(gray, out_dir / "caddy.bmp");

	GrayImage dst_gray;
	img::make_image(dst_gray, gray.width, gray.height);

	img::transform(gray, dst_gray, invert);
	img::write_image(dst_gray, out_dir / "transform_gray.bmp");

	clear_image(dst_gray);

	img::seq::transform(gray, dst_gray, invert);
	img::write_image(dst_gray, out_dir / "seq_transform_gray.bmp");
}


void alpha_blend_test()
{
	auto title = "alpha_blend_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	Image caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);

	Image cur;
	cur.width = width;
	cur.height = height;
	img::resize_image(caddy, cur);

	Image dst;
	img::make_image(dst, width, height);

	img::seq::transform_alpha(src, [](auto const& p) { return 128; });

	img::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "alpha_blend.bmp");

	clear_image(dst);

	img::seq::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "seq_alpha_blend.bmp");

	/*img::simd::alpha_blend(src, cur, dst);
	img::write_image(dst, out_dir / "simd_alpha_blend.bmp");*/

	img::seq::copy(cur, dst);
	img::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "alpha_blend_src_dst.bmp");

	img::seq::copy(cur, dst);
	img::seq::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "seq_alpha_blend_src_dst.bmp");

	/*img::seq::copy(cur, dst);
	img::simd::alpha_blend(src, dst);
	img::write_image(dst, out_dir / "simd_alpha_blend_src_dst.bmp");*/

}


void grayscale_test()
{
	auto title = "grayscale_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image src;
	img::read_image_from_file(CORVETTE_PATH, src);
	auto width = src.width;
	auto height = src.height;

	GrayImage dst;
	img::make_image(dst, width, height);

	img::grayscale(src, dst);
	img::write_image(dst, out_dir / "grayscale.bmp");

	clear_image(dst);

	img::seq::grayscale(src, dst);
	img::write_image(dst, out_dir / "seq_grayscale.bmp");

	/*img::simd::grayscale(src, dst);
	img::write_image(dst, out_dir / "simd_grayscale.bmp");*/
}


void binary_test()
{
	auto title = "binary_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	GrayImage caddy;
	img::read_image_from_file(CADILLAC_PATH, caddy);
	img::write_image(caddy, out_dir / "caddy.bmp");

	GrayImage caddy_binary;
	img::make_image(caddy_binary, caddy.width, caddy.height);

	img::binarize_th(caddy, caddy_binary, 128);
	img::write_image(caddy_binary, out_dir / "caddy_binary.bmp");

	Image weed;
	img::read_image_from_file(WEED_PATH, weed);
	auto width = weed.width;
	auto height = weed.height;

	GrayImage binary_src;
	img::make_image(binary_src, width, height);

	GrayImage binary_dst;
	img::make_image(binary_dst, width, height);

	GrayImage temp;
	img::make_image(temp, width, height);

	auto const is_white = [](Pixel p)
	{
		return ((r32)p.red + (r32)p.blue + (r32)p.green) / 3.0f < 190;
	};

	img::binarize(weed, binary_src, is_white);
	img::write_image(binary_src, out_dir / "weed.bmp");

	// centroid point
	auto pt = img::centroid(binary_src);

	// region around centroid
	img::pixel_range_t c{};
	c.x_begin = pt.x - 10;
	c.x_end = pt.x + 10;
	c.y_begin = pt.y - 10;
	c.y_end = pt.y + 10;

	// draw binary image with centroid region
	img::copy(binary_src, binary_dst);
	auto c_view = img::sub_view(binary_dst, c);
	std::fill(c_view.begin(), c_view.end(), 0);
	img::write_image(binary_dst, out_dir / "centroid.bmp");

	// thin the object
	img::seq::skeleton(binary_src, binary_dst);
	img::write_image(binary_dst, out_dir / "skeleton.bmp");
}


void contrast_test()
{
	auto title = "contrast_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	img::write_image(src, out_dir / "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::contrast(src, dst, 0, 128);
	img::write_image(dst, out_dir / "contrast.bmp");

	clear_image(dst);

	img::seq::contrast(src, dst, 0, 128);
	img::write_image(dst, out_dir / "seq_contrast.bmp");
}


void blur_test()
{
	auto title = "blur_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	img::write_image(src, out_dir / "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::blur(src, dst);
	img::write_image(dst, out_dir / "blur.bmp");

	img::seq::blur(src, dst);
	img::write_image(dst, out_dir / "seq_blur.bmp");

	/*img::simd::blur(src, dst);
	img::write_image(dst, out_dir / "simd_blur.bmp");*/
}


void gradients_test()
{
	auto title = "gradients_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	img::write_image(src, out_dir / "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	img::gradients(src, dst);
	img::write_image(dst, out_dir / "gradient.bmp");

	clear_image(dst);

	img::seq::gradients(src, dst);
	img::write_image(dst, out_dir / "seq_gradient.bmp");

	/*img::simd::gradients(src, dst);
	img::write_image(dst, out_dir / "simd_gradient.bmp");*/
}


void edges_test()
{
	auto title = "edges_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
	img::write_image(src, out_dir / "vette.bmp");

	GrayImage dst;
	img::make_image(dst, src.width, src.height);

	auto const threshold = [](u8 g) { return g >= 100; };

	img::edges(src, dst, threshold);
	img::write_image(dst, out_dir / "edges.bmp");

	clear_image(dst);

	img::seq::edges(src, dst, threshold);
	img::write_image(dst, out_dir / "seq_edges.bmp");

	/*img::simd::edges(src, dst, threshold);
	img::write_image(dst, out_dir / "simd_edges.bmp");*/
}


void combo_view_test()
{
    auto title = "combo_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

    GrayImage src;
	img::read_image_from_file(CORVETTE_PATH, src);
    auto width = src.width;
    auto height = src.height;

    GrayImage dst;
    img::make_image(dst, width, height);

    // top left
    img::pixel_range_t range{};
    range.x_begin = 0;
	range.x_end = width / 2;
	range.y_begin = 0;
	range.y_end = height / 2;
    auto src_sub = img::sub_view(src, range);
	auto dst_sub = img::sub_view(dst, range);

	img::seq::gradients(src_sub, dst_sub);

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

    img::seq::binarize_th(src_sub, dst_sub, 75);

    // bottom right
    range.x_begin = width / 2;
	range.x_end = width;
	src_sub = img::sub_view(src, range);
	dst_sub = img::sub_view(dst, range);

    auto const threshold = [](u8 g) { return g >= 100; };
    img::seq::edges(src_sub, dst_sub, threshold);

    img::write_image(dst, out_dir / "combo.bmp");
}


void rotate_test()
{
	auto title = "rotate_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);

	Image src;
	img::read_image_from_file(CADILLAC_PATH, src);

	Image dst;
	img::make_image(dst, src.width, src.height);

	Point2Du32 origin = { src.width / 2, src.height / 2 };

	r32 theta = 0.6f * 2 * 3.14159f;
	img::seq::rotate(src, dst, origin, theta);
	img::write_image(dst, out_dir / "rotate.bmp");

	GrayImage src_gray;
	img::read_image_from_file(CORVETTE_PATH, src_gray);

	GrayImage dst_gray;
	img::make_image(dst_gray, src_gray.width, src_gray.height);

	origin = { src_gray.width / 2, src_gray.height / 2 };

	img::seq::rotate(src_gray, dst_gray, origin, theta);
	img::write_image(dst_gray, out_dir / "rotate_gray.bmp");
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
	std::fill(img.begin(), img.end(), black);
}


void clear_image(GrayImage const& img)
{
	std::fill(img.begin(), img.end(), 0);
}