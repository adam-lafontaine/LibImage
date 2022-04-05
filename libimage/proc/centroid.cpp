#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


#ifndef LIBIMAGE_NO_GRAYSCALE

	Point2Du32 centroid(gray::image_t const& src)
	{
		auto const func = [](u8 p) { return p > 0; };
		return centroid(src, func);
	}


	Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		u32 const n_threads = 10;
		u32 h = src.height / n_threads;

		u32 total = 0;
		u32 x_total = 0;
		u32 y_total = 0;

		auto const func = [&](u32 i)
		{
			auto y_begin = i * h;
			auto y_end = (i + 1) * h;

			for (u32 y = y_begin; y < src.height && y < y_end; ++y)
			{
				auto row = src.row_begin(y);
				for (u32 x = 0; x < src.width; ++x)
				{
					u32 val = func(*src.xy_at(x, y)) ? 1 : 0;

					total += val;
					x_total += x * val;
					y_total += y * val;
				}
			}
		};

		u32_range_t ids(n_threads);

		std::for_each(std::execution::par, ids.begin(), ids.end(), func);

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = x_total / total;
			pt.y = y_total / total;
		}

		return pt;
	}



#endif // !LIBIMAGE_NO_GRAYSCALE


#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		Point2Du32 centroid(gray::image_t const& src)
		{
			auto const func = [](u8 p) { return p > 0; };
			return centroid(src, func);
		}


		Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
		{
			assert(verify(src));

			u32 total = 0;
			u32 x_total = 0;
			u32 y_total = 0;

			auto const xy_func = [&](u32 x, u32 y)
			{
				u32 val = func(*src.xy_at(x, y)) ? 1 : 0;

				total += val;
				x_total += x * val;
				y_total += y * val;
			};

			for_each_xy(src, xy_func);

			Point2Du32 pt{};

			if (total == 0)
			{
				pt.x = src.width / 2;
				pt.y = src.height / 2;
			}
			else
			{
				pt.x = x_total / total;
				pt.y = y_total / total;
			}

			return pt;
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}