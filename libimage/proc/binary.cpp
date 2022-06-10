#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
#include "verify.hpp"
#include "index_range.hpp"

#include <array>
#include <algorithm>
#include <numeric>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

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
		constexpr u32 n_threads = 20;
		u32 h = src.height / n_threads;

		std::array<u32, n_threads> thread_totals = { 0 };
		std::array<u32, n_threads> thread_x_totals = { 0 };
		std::array<u32, n_threads> thread_y_totals = { 0 };

		u32 total = 0;
		u32 x_total = 0;
		u32 y_total = 0;

		auto const row_func = [&](u32 y) 
		{
			if (y >= src.height)
			{
				return;
			}

			auto thread_id = y - n_threads * (y / n_threads);

			assert(thread_id < n_threads);

			auto row = src.row_begin(y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u32 val = func(row[x]) ? 1 : 0;

				thread_totals[thread_id] += val;
				thread_x_totals[thread_id] += x * val;
				thread_y_totals[thread_id] += y * val;
			}
		};

		for (u32 y_begin = 0; y_begin < src.height; y_begin += n_threads)
		{
			thread_totals = { 0 };
			thread_x_totals = { 0 };
			thread_y_totals = { 0 };

			u32_range_t rows(y_begin, y_begin + n_threads);

			std::for_each(std::execution::par, rows.begin(), rows.end(), row_func);

			for (u32 i = 0; i < n_threads; ++i)
			{
				total += thread_totals[i];
				x_total += thread_x_totals[i];
				y_total += thread_y_totals[i];
			}
		}

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

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
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


		template <class GRAY_IMG_T>
		static bool do_neighbors(GRAY_IMG_T const& img, u32 x, u32 y)
		{
			assert(x >= 1);
			assert(x < img.width);
			assert(y >= 1);
			assert(y < img.height);

			constexpr std::array<int, 8> x_neighbors = { -1,  0,  1,  1,  1,  0, -1, -1 };
			constexpr std::array<int, 8> y_neighbors = { -1, -1, -1,  0,  1,  1,  1,  0 };

			constexpr auto n_neighbors = x_neighbors.size();
			int value_total = 0;
			u32 value_count = 0;
			u32 flip = 0;

			auto xi = (u32)(x + x_neighbors[n_neighbors - 1]);
			auto yi = (u32)(y + y_neighbors[n_neighbors - 1]);
			auto val = *img.xy_at(xi, yi);
			bool is_on = val != 0;

			for (u32 i = 0; i < n_neighbors; ++i)
			{
				xi = (u32)(x + x_neighbors[i]);
				yi = (u32)(y + y_neighbors[i]);

				val = *img.xy_at(xi, yi);
				flip += (val != 0) != is_on;

				is_on = val != 0;
				value_count += is_on;
			}

			return value_count > 1 && value_count < 7 && flip == 2;
		}


		template <class GRAY_IMG_T>
		static u32 skeleton_once(GRAY_IMG_T const& img)
		{
			u32 pixel_count = 0;

			auto width = img.width;
			auto height = img.height;

			auto const xy_func = [&](u32 x, u32 y) 
			{				
				auto& p = *img.xy_at(x, y);
				if (p == 0)
				{
					return;
				}

				if (do_neighbors(img, x, y))
				{
					p = 0;
				}

				pixel_count += p > 0;
			};

			u32 x_begin = 1;
			u32 x_end = width - 1;
			u32 y_begin = 1;
			u32 y_end = height - 2;
			u32 x = 0;
			u32 y = 0;

			auto const done = [&]() { return !(x_begin < x_end && y_begin < y_end); };

			while (!done())
			{
				// iterate clockwise
				y = y_begin;
				x = x_begin;
				for (; x < x_end; ++x)
				{
					xy_func(x, y);
				}
				--x;
				
				for (++y; y < y_end; ++y)
				{
					xy_func(x, y);
				}
				--y;

				for (--x; x >= x_begin; --x)
				{
					xy_func(x, y);
				}
				++x;

				for (--y; y > y_begin; --y)
				{
					xy_func(x, y);
				}
				++y;
				
				++x_begin;
				++y_begin;
				--x_end;
				--y_end;

				if (done())
				{
					break;
				}

				// iterate counter clockwise
				for (++x; y < y_end; ++y)
				{
					xy_func(x, y);
				}
				--y;

				for (++x; x < x_end; ++x)
				{
					xy_func(x, y);
				}
				--x;

				for (--y; y >= y_begin; --y)
				{
					xy_func(x, y);
				}
				++y;

				for (--x; x >= x_begin; --x)
				{
					xy_func(x, y);
				}
				++x;

				++x_begin;
				++y_begin;
				--x_end;
				--y_end;
			}

			return pixel_count;
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_skeleton(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::copy(src, dst);

			u32 current_count = 0;
			u32 pixel_count = skeleton_once(dst);
			u32 max_iter = 100; // src.width / 2;

			for (u32 i = 1; pixel_count != current_count && i < max_iter; ++i)
			{
				current_count = pixel_count;
				pixel_count = skeleton_once(dst);
			}
		}


		void skeleton(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			do_skeleton(src, dst);
		}


		void skeleton(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			do_skeleton(src, dst);
		}


		void skeleton(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			do_skeleton(src, dst);
		}


		void skeleton(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			do_skeleton(src, dst);
		}

	}

}

#endif // !LIBIMAGE_NO_GRAYSCALE