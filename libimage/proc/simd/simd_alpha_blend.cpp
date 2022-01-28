/*

#ifndef LIBIMAGE_NO_SIMD

#include "../verify.hpp"
#include "simd_def.hpp"

namespace libimage
{
	namespace simd
	{

	}
}

#endif // !LIBIMAGE_NO_SIMD

*/

#ifndef LIBIMAGE_NO_SIMD

#include "../verify.hpp"
#include "simd_def.hpp"

namespace libimage
{
	namespace simd
	{
		static void alpha_blend_row(pixel_t* src_begin, pixel_t* cur_begin, pixel_t* dst_begin, u32 length)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			r32 one = 1.0f;
			r32 u8max = 255.0f;

			auto const do_simd = [&](u32 i)
			{
				// pixels are interleaved
				// make them planar
				PixelPlanar src_mem{};
				PixelPlanar cur_mem{};
				PixelPlanar dst_mem{};

				auto src = src_begin + i;
				auto cur = cur_begin + i;
				auto dst = dst_begin + i;

				for (u32 j = 0; j < N; ++j)
				{
					src_mem.red[j] = (r32)src[j].red;
					src_mem.green[j] = (r32)src[j].green;
					src_mem.blue[j] = (r32)src[j].blue;
					src_mem.alpha[j] = (r32)src[j].alpha;

					cur_mem.red[j] = (r32)cur[j].red;
					cur_mem.green[j] = (r32)cur[j].green;
					cur_mem.blue[j] = (r32)cur[j].blue;
				}

				auto one_vec = simd_load_broadcast(&one);
				auto u8max_vec = simd_load_broadcast(&u8max);

				auto src_a_vec = simd_divide(simd_load(src_mem.alpha), u8max_vec);
				auto cur_a_vec = simd_subtract(one_vec, src_a_vec);

				auto dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.red), simd_multiply(cur_a_vec, simd_load(cur_mem.red)));
				simd_store(dst_mem.red, dst_vec);

				dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.green), simd_multiply(cur_a_vec, simd_load(cur_mem.green)));
				simd_store(dst_mem.green, dst_vec);

				dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.blue), simd_multiply(cur_a_vec, simd_load(cur_mem.blue)));
				simd_store(dst_mem.blue, dst_vec);

				for (u32 j = 0; j < N; ++j)
				{
					dst[j].red =   (u8)dst_mem.red[j];
					dst[j].green = (u8)dst_mem.green[j];
					dst[j].blue =  (u8)dst_mem.blue[j];
					dst[j].alpha = 255;
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		void alpha_blend(image_t const& src, image_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			alpha_blend_row(src.begin(), current.begin(), dst.begin(), src.width * src.height);
		}


		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(image_t const& src, image_t const& current_dst)
		{
			assert(verify(src, current_dst));

			alpha_blend_row(src.begin(), current_dst.begin(), current_dst.begin(), src.width * src.height);
		}


		void alpha_blend(image_t const& src, view_t const& current_dst)
		{
			assert(verify(src, current_dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current_dst.row_begin(y), current_dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, image_t const& current_dst)
		{
			assert(verify(src, current_dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current_dst.row_begin(y), current_dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(view_t const& src, view_t const& current_dst)
		{
			assert(verify(src, current_dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_blend_row(src.row_begin(y), current_dst.row_begin(y), current_dst.row_begin(y), src.width);
			}
		}


		void alpha_blend(image_soa const& src, image_soa const& cur, image_soa const& dst)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			r32 one = 1.0f;
			r32 u8max = 255.0f;

			auto const do_simd = [&](u32 i)
			{
				PixelPlanar src_mem{};
				PixelPlanar cur_mem{};
				PixelPlanar dst_mem{};

				for (u32 j = 0; j < N; ++j)
				{
					src_mem.red[j]   = (r32)(src.red + i)[j];
					src_mem.green[j] = (r32)(src.green + i)[j];
					src_mem.blue[j]  = (r32)(src.blue + i)[j];
					src_mem.alpha[j] = (r32)(src.alpha + i)[j];

					cur_mem.red[j]   = (r32)(cur.red + i)[j];
					cur_mem.green[j] = (r32)(cur.green + i)[j];
					cur_mem.blue[j]  = (r32)(cur.blue + i)[j];
					cur_mem.alpha[j] = (r32)(cur.alpha + i)[j];

					dst_mem.red[j]   = (r32)(dst.red + i)[j];
					dst_mem.green[j] = (r32)(dst.green + i)[j];
					dst_mem.blue[j]  = (r32)(dst.blue + i)[j];
				}

				auto one_vec = simd_load_broadcast(&one);
				auto u8max_vec = simd_load_broadcast(&u8max);

				auto src_a_vec = simd_divide(simd_load(src_mem.alpha), u8max_vec);
				auto cur_a_vec = simd_subtract(one_vec, src_a_vec);

				auto dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.red), simd_multiply(cur_a_vec, simd_load(cur_mem.red)));
				simd_store(dst_mem.red, dst_vec);

				dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.green), simd_multiply(cur_a_vec, simd_load(cur_mem.green)));
				simd_store(dst_mem.green, dst_vec);

				dst_vec = simd_fmadd(src_a_vec, simd_load(src_mem.blue), simd_multiply(cur_a_vec, simd_load(cur_mem.blue)));
				simd_store(dst_mem.blue, dst_vec);

				for (u32 j = 0; j < N; ++j)
				{
					(dst.red + i)[j]   = (u8)dst_mem.red[j];
					(dst.green + i)[j] = (u8)dst_mem.green[j];
					(dst.blue + i)[j]  = (u8)dst_mem.blue[j];
					(dst.alpha + i)[j] = 255;
				}
			};

			auto length = src.width * src.height;

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		void alpha_blend(image_soa const& src, image_soa const& current_dst)
		{
			simd::alpha_blend(src, current_dst, current_dst);
		}
	}
}

#endif // !LIBIMAGE_NO_SIMD