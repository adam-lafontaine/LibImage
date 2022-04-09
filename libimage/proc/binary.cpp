#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <array>
#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}


	void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}


#ifndef LIBIMAGE_NO_COLOR

	void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)

	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


#endif // !LIBIMAGE_NO_COLOR


	template <class GRAY_IMG_T>
	Point2Du32 do_centroid(GRAY_IMG_T const& src, u8_to_bool_f const& func)
	{
		u32 const n_threads = 10;
		u32 h = src.height / n_threads;

		u32 total = 0;
		u32 x_total = 0;
		u32 y_total = 0;

		auto const xy_func = [&](u32 i)
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

		std::for_each(std::execution::par, ids.begin(), ids.end(), xy_func);

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


	Point2Du32 centroid(gray::image_t const& src)
	{
		assert(verify(src));

		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::view_t const& src)
	{
		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform_self(src_dst, conv);
		}


		void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform_self(src_dst, conv);
		}


#ifndef LIBIMAGE_NO_COLOR

		void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


#endif // !LIBIMAGE_NO_COLOR


		template <class GRAY_IMG_T>
		Point2Du32 do_centroid(GRAY_IMG_T const& src, u8_to_bool_f const& func)
		{
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


		Point2Du32 centroid(gray::image_t const& src)
		{
			assert(verify(src));

			auto const func = [](u8 p) { return p > 0; };
			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
		{
			assert(verify(src));

			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::view_t const& src)
		{
			assert(verify(src));

			auto const func = [](u8 p) { return p > 0; };
			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func)
		{
			assert(verify(src));

			return seq::do_centroid(src, func);
		}


		template<class GRAY_IMG_T>
		static void zero_top_bottom(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 1;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = 0;
				dst_bottom[x] = 0;
			}
		}


		template<class GRAY_IMG_T>
		static void zero_left_right(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 1;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = 0;
				dst_row[x_last] = 0;
			}
		}


		template <class GRAY_IMG_T>
		static bool neighbors_connected(GRAY_IMG_T const& img, u32 x, u32 y)
		{
			assert(x >= 1);
			assert(x < img.width);
			assert(y >= 1);
			assert(y < img.height);

			constexpr std::array<int, 4> x_neighbors = { 0, 1, 0, -1 };
			constexpr std::array<int, 4> y_neighbors = { -1, 0, 1,  0 };
			constexpr std::array<int, 4> values = { 1, 2, 4, 8 };

			//                                              0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15
			constexpr std::array<u32, 16> value_results = { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,  0,  1,  1,  1,  1,  1 };

			auto const n_neighbors = x_neighbors.size();
			int value_total = 0;

			for (u32 i = 0; i < n_neighbors; ++i)
			{
				auto xi = (u32)(x + x_neighbors[i]);
				auto yi = (u32)(y + y_neighbors[i]);

				auto val = *img.xy_at(xi, yi);
				if (val > 0)
				{
					value_total += values[i];
				}
			}

			if (value_total < 0 || value_total > 15)
			{
				value_total = 0;
			}

			return value_results[value_total] > 0;
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static bool thin_once(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			bool equal = true;

			zero_top_bottom(dst);
			zero_left_right(dst);

			for (u32 y = 1; y < src.height - 1; ++y)
			{
				auto src_row = src.row_begin(y);
				auto dst_row = dst.row_begin(y);
				for (u32 x = 1; x < src.width - 1; ++x)
				{
					auto p = src_row[x];
					dst_row[x] = (p == 0 || neighbors_connected(src, x, y)) ? 0 : p;

					equal &= dst_row[x] == p;
				}
			}

			return equal;
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_thin(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, gray::image_t const& temp)
		{
			bool equal = thin_once(src, dst);
			u32 max_iter = src.width;

			for (u32 i = 0; !equal && i < max_iter; ++i)
			{
				thin_once(dst, temp);
				equal = thin_once(temp, dst);
			}
		}


		void thin_objects(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			do_thin(src, dst, temp);
		}


		void thin_objects(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			do_thin(src, dst, temp);
		}


		void thin_objects(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			do_thin(src, dst, temp);
		}


		void thin_objects(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			do_thin(src, dst, temp);
		}



#endif // !LIBIMAGE_NO_GRAYSCALE
	}

}