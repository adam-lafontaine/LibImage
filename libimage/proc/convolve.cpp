/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "convolve.hpp"
#include "../libimage.hpp"

#include <array>

#ifndef LIBIMAGE_NO_SIMD
#include <xmmintrin.h>
#endif // !LIBIMAGE_NO_SIMD



namespace libimage
{
	static inline void left_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left (2 wide)
		assert(x <= width - 2);
		range.x_begin = x;
		range.x_end = x + 2;
	}


	static inline void right_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// right (2 wide)
		assert(x >= 1);
		assert(x <= width - 1);
		range.x_begin = x - 1;
		range.x_end = x + 1;
	}


	static inline void top_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top (2 high)
		assert(y <= height - 2);
		range.y_begin = y;
		range.y_end = y + 2;
	}


	static inline void bottom_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// bottom (2 high)
		assert(y >= 1);
		assert(y <= height - 1);
		range.y_begin = y - 1;
		range.y_end = y + 1;
	}


	static inline void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (3 high)
		assert(y >= 1);
		assert(y <= height - 2);
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


	static inline void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (3 wide)
		assert(x >= 1);
		assert(x <= width - 2);
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


	static inline void top_or_bottom_5_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (5 high)
		assert(y >= 2);
		assert(y <= height - 3);
		range.y_begin = y - 2;
		range.y_end = y + 3;
	}


	static inline void left_or_right_5_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (5 wide)
		assert(x >= 2);
		assert(x <= width - 3);
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


	template<size_t N>
	static r32 apply_weights(gray::view_t const& view, std::array<r32, N> weights)
	{
		assert(static_cast<size_t>(view.width) * view.height == weights.size());

		u32 w = 0;
		r32 total = 0.0f;

		auto const add_weight = [&](u8 p) 
		{ 
			total += weights[w++] * p; 
		};

		for_each_pixel(view, add_weight);

		return total;
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_5_high(range, y, img.height);

		left_or_right_5_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	constexpr r32 D3 = 16.0f;
	constexpr std::array<r32, 9> GAUSS_3X3
	{
		(1 / D3), (2 / D3), (1 / D3),
		(2 / D3), (4 / D3), (2 / D3),
		(1 / D3), (2 / D3), (1 / D3),
	};

	constexpr r32 D5 = 256.0f;
	constexpr std::array<r32, 25> GAUSS_5X5
	{
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
	};


	constexpr std::array<r32, 9> GRAD_X_3X3
	{
		1.0f, 0.0f, -1.0f,
		2.0f, 0.0f, -2.0f,
		1.0f, 0.0f, -1.0f,
	};
	constexpr std::array<r32, 9> GRAD_Y_3X3
	{
		1.0f,  2.0f,  1.0f,
		0.0f,  0.0f,  0.0f,
		-1.0f, -2.0f, -1.0f,
	};


	u8 gauss3(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss3(gray::view_t const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss5(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	u8 gauss5(gray::view_t const& view, u32 x, u32 y)
	{		
		auto p = weighted_center(view, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return static_cast<u8>(p);
	}


	r32 x_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_X_3X3);
	}


	r32 x_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_X_3X3);
	}


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_Y_3X3);
	}


	r32 y_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_Y_3X3);
	}

#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void copy_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			constexpr u8 N = 4;
			constexpr u32 STEP = N * sizeof(32) / sizeof(u8);

			u32 x_begin = 0;
			u32 x_end = src.width;
			u32 y_begin = 0;
			u32 y_last = src.height - 1;

			auto src_top = src.row_begin(y_begin);
			auto src_bottom = src.row_begin(y_last);
			auto dst_top = dst.row_begin(y_begin);
			auto dst_bottom = dst.row_begin(y_last);

			auto length = x_end - x_begin;

			auto const do_simd = [&](u32 offset_u8)
			{
				auto src_top32 = (r32*)(src_top + offset_u8);
				auto dst_top32 = (r32*)(dst_top + offset_u8);
				auto src_bottom32 = (r32*)(src_bottom + offset_u8);
				auto dst_bottom32 = (r32*)(dst_bottom + offset_u8);

				_mm_store_ps(dst_top32, _mm_load_ps(src_top32));
				_mm_store_ps(dst_bottom32, _mm_load_ps(src_bottom32));
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		static void gauss3_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;
			r32 memory[N];

			auto const do_simd = [&](int i)
			{
				u32 w = 0;
				auto acc_vec = _mm_setzero_ps();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;

						for (u32 n = 0; n < N; ++n)
						{
							memory[n] = static_cast<r32>(src_begin[offset + (int)n]);
						}

						auto src_vec = _mm_load_ps(memory);

						auto weight = _mm_load1_ps(GAUSS_3X3.data() + w);

						acc_vec = _mm_add_ps(acc_vec, _mm_mul_ps(weight, src_vec));
					}
				}

				_mm_store_ps(memory, acc_vec);

				for (u32 n = 0; n < N; ++n)
				{
					dst_begin[i + n] = static_cast<u8>(memory[n]);
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_begin = 1;
			u32 y_last = src.height - 2;

			auto length = x_end - x_begin;
			auto pitch = static_cast<u32>(src.row_begin(1) - src.row_begin(0));

			auto src_row = src.row_begin(y_begin) + x_begin;
			auto dst_row = dst.row_begin(y_begin) + x_begin;
			gauss3_row(src_row, dst_row, length, pitch);

			src_row = src.row_begin(y_last) + x_begin;
			dst_row = dst.row_begin(y_last) + x_begin;
			gauss3_row(src_row, dst_row, length, pitch);
		}


		static void gauss5_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;
			r32 memory[N];

			auto const do_simd = [&](int i)
			{
				u32 w = 0;
				auto acc_vec = _mm_setzero_ps();

				for (int ry = -2; ry < 3; ++ry)
				{
					for (int rx = -2; rx < 3; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;

						for (u32 n = 0; n < N; ++n)
						{
							memory[n] = static_cast<r32>(src_begin[offset + (int)n]);
						}

						auto src_vec = _mm_load_ps(memory);

						auto weight = _mm_load1_ps(GAUSS_5X5.data() + w);

						acc_vec = _mm_add_ps(acc_vec, _mm_mul_ps(weight, src_vec));
					}
				}

				_mm_store_ps(memory, acc_vec);

				for (u32 n = 0; n < N; ++n)
				{
					dst_begin[i + n] = static_cast<u8>(memory[n]);
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void gauss5_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_begin = 2;
			u32 y_begin = 2;
			u32 x_end = src.width - 2;
			u32 y_end = src.height - 2;

			auto length = x_end - x_begin;
			auto pitch = static_cast<u32>(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto src_row = src.row_begin(y) + x_begin;
				auto dst_row = dst.row_begin(y) + x_begin;
				gauss5_row(src_row, dst_row, length, pitch);
			}
		}


		static void xy_gradients_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;
			r32 memory[N];

			auto const do_simd = [&](int i)
			{
				u32 w = 0;
				auto vec_x = _mm_setzero_ps();
				auto vec_y = _mm_setzero_ps();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;

						for (u32 n = 0; n < N; ++n)
						{
							memory[n] = static_cast<r32>(src_begin[offset + (int)n]);
						}

						auto src_vec = _mm_load_ps(memory);

						auto weight_x = _mm_load1_ps(GRAD_X_3X3.data() + w);
						auto weight_y = _mm_load1_ps(GRAD_Y_3X3.data() + w);

						vec_x = _mm_add_ps(vec_x, _mm_mul_ps(weight_x, src_vec));
						vec_y = _mm_add_ps(vec_y, _mm_mul_ps(weight_y, src_vec));
					}
				}

				vec_x = _mm_mul_ps(vec_x, vec_x);
				vec_y = _mm_mul_ps(vec_y, vec_y);

				auto grad = _mm_sqrt_ps(_mm_add_ps(vec_x, vec_y));

				_mm_store_ps(memory, grad);

				for (u32 n = 0; n < N; ++n)
				{
					dst_begin[i + n] = static_cast<u8>(memory[n]);
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void xy_gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_begin = 1;
			u32 y_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_end = src.height - 1;

			auto length = x_end - x_begin;
			auto pitch = static_cast<u32>(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				xy_gradients_row(src.row_begin(y) + x_begin, dst.row_begin(y) + x_begin, length, pitch);
			}
		}


		static void edges_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch, u8_to_bool_f const& cond)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;
			r32 memory[N];

			auto const do_simd = [&](int i)
			{
				u32 w = 0;
				auto vec_x = _mm_setzero_ps();
				auto vec_y = _mm_setzero_ps();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;

						for (u32 n = 0; n < N; ++n)
						{
							memory[n] = static_cast<r32>(src_begin[offset + (int)n]);
						}

						auto src_vec = _mm_load_ps(memory);

						auto weight_x = _mm_load1_ps(GRAD_X_3X3.data() + w);
						auto weight_y = _mm_load1_ps(GRAD_Y_3X3.data() + w);

						vec_x = _mm_add_ps(vec_x, _mm_mul_ps(weight_x, src_vec));
						vec_y = _mm_add_ps(vec_y, _mm_mul_ps(weight_y, src_vec));
					}
				}

				vec_x = _mm_mul_ps(vec_x, vec_x);
				vec_y = _mm_mul_ps(vec_y, vec_y);

				auto grad = _mm_sqrt_ps(_mm_add_ps(vec_x, vec_y));

				_mm_store_ps(memory, grad);

				for (u32 n = 0; n < N; ++n)
				{
					dst_begin[i + n] = cond(static_cast<u8>(memory[n])) ? 255 : 0;
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			u32 x_begin = 1;
			u32 y_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_end = src.height - 1;

			auto length = x_end - x_begin;
			auto pitch = static_cast<u32>(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				edges_row(src.row_begin(y) + x_begin, dst.row_begin(y) + x_begin, length, pitch, cond);
			}
		}


		void inner_gauss(gray::image_t const& src, gray::image_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::image_t const& src, gray::view_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::view_t const& src, gray::image_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::view_t const& src, gray::view_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}
		
	}

#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_GRAYSCALE