#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <cmath>
#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Point2Dr32
{
public:
	r32 x;
	r32 y;
};


static Point2Dr32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, r32 theta_rotate)
{
	auto dx_dst = (r32)pt.x - (r32)origin.x;
	auto dy_dst = (r32)pt.y - (r32)origin.y;

	auto radius = std::hypotf(dx_dst, dy_dst);

	auto theta_dst = atan2f(dy_dst, dx_dst);
	auto theta_src = theta_dst - theta_rotate;

	auto dx_src = radius * cosf(theta_src);
	auto dy_src = radius * sinf(theta_src);

	Point2Dr32 pt_src{};
	pt_src.x = (r32)origin.x + dx_src;
	pt_src.y = (r32)origin.y + dy_src;

	return pt_src;
}





namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	template <typename IMG_T>
	static pixel_t get_color(IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return to_pixel(0, 0, 0);
		}

		return *src_image.xy_at((u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_T>
	static u8 get_gray(GR_IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return 0;
		}

		return *src_image.xy_at((u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	


	template <typename IMG_SRC_T, typename IMG_DST_T>
	static void rotate_par(IMG_SRC_T const& src, IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		u32_range_t range(dst.width * dst.height);

		auto const func = [&](u32 i)
		{
			auto y = i / dst.width;
			auto x = i - y * dst.width;
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*dst.xy_at(x, y) = get_color(src, src_pt);
		};

		std::for_each(std::execution::par, range.begin(), range.end(), func);
	}


	void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_SRC_T, typename GR_IMG_DST_T>
	static void rotate_par_gray(GR_IMG_SRC_T const& src, GR_IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		u32_range_t range(dst.width * dst.height);

		auto const func = [&](u32 i)
		{
			auto y = i / dst.width;
			auto x = i - y * dst.width;
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*dst.xy_at(x, y) = get_gray(src, src_pt);
		};

		std::for_each(std::execution::par, range.begin(), range.end(), func);
	}


	void rotate(gray::image_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::image_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::view_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::view_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR


		template <typename IMG_SRC_T, typename IMG_DST_T>
		static void rotate_seq(IMG_SRC_T const& src, IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			Point2Du32 origin = { origin_x, origin_y };

			auto const func = [&](u32 x, u32 y)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, theta);
				*dst.xy_at(x, y) = get_color(src, src_pt);
			};

			for_each_xy(dst, func);
		}


		void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		template <typename GR_IMG_SRC_T, typename GR_IMG_DST_T>
		static void rotate_seq_gray(GR_IMG_SRC_T const& src, GR_IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			Point2Du32 origin = { origin_x, origin_y };

			auto const func = [&](u32 x, u32 y)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, theta);
				*dst.xy_at(x, y) = get_gray(src, src_pt);
			};

			for_each_xy(dst, func);
		}


		void rotate(gray::image_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::image_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::view_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::view_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}