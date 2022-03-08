#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <algorithm>
#include <cmath>

#include "proc_def.hpp"

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


static Point2Dr32 find_rotation_src(Point2Du32 const& pt_dst, Point2Du32 const& origin, r32 theta_rotate)
{
	auto dx_dst = (r32)pt_dst.x - (r32)origin.x;
	auto dy_dst = (r32)pt_dst.y - (r32)origin.y;

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
	static u8 blend4(u8 cxy, u8 cx1y, u8 cxy1, u8 cx1y1, r32 fx, r32 fy)
	{
		auto gx = 1.0f - fx;
		auto gy = 1.0f - fy;

		auto blend =
			gx * gy * cxy +
			fx * gy * cx1y +
			gx * fy * cxy1 +
			fx * fy * cx1y1;

		return (u8)roundf(blend);
	}


	static pixel_t get_color(image_t const& src_image, Point2Dr32 location)
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

		auto floorx = floorf(x);
		auto floory = floorf(y);

		auto fx = x - floorx;
		auto fy = y = floory;

		auto x0 = (u32)floorx;
		auto x1 = x0 + 1;
		auto y0 = (u32)floory;
		auto y1 = y0 + 1;		

		auto pxy = src_image.xy_at(x0, y0);
		auto px1y = src_image.xy_at(x1, y0);
		auto pxy1 = src_image.xy_at(x0, y1);
		auto px1y1 = src_image.xy_at(x1, y1);

		auto r = blend4(pxy->red, px1y->red, pxy1->red, px1y1->red, fx, fy);
		auto g = blend4(pxy->green, px1y->green, pxy1->green, px1y1->green, fx, fy);
		auto b = blend4(pxy->blue, px1y->blue, pxy1->blue, px1y1->blue, fx, fy);

		return to_pixel(r, g, b);
	}


	void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 rad)
	{
		assert(verify(src));
		assert(verify(dst));

		Point2Du32 origin = { origin_x, origin_y };

		auto const func = [&](u32 x, u32 y) 
		{
			auto src_pt = find_rotation_src({ x, y }, origin, rad);
			*dst.xy_at(x, y) = get_color(src, src_pt);
		};

		for_each_xy(dst, func);
	}
}