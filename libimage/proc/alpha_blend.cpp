#ifndef LIBIMAGE_NO_COLOR

#include "process.hpp"
#include "verify.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_SIMD
#include <xmmintrin.h>
#include <immintrin.h>
#endif // !LIBIMAGE_NO_SIMD

namespace libimage
{
	static pixel_t alpha_blend_linear(pixel_t const& src, pixel_t const& current)
	{
		auto const a = (r32)(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = (r32)(s);
			auto cf = (r32)(c);

			auto blended = a * cf + (1.0f - a) * sf;

			return (u8)(blended);
		};

		auto red = blend(src.red, current.red);
		auto green = blend(src.green, current.green);
		auto blue = blend(src.blue, current.blue);

		return to_pixel(red, green, blue);
	}


#ifndef LIBIMAGE_NO_PARALLEL

	void alpha_blend(image_t const& src, image_t const& current, image_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(image_t const& src, image_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(image_t const& src, view_t const& current, image_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(image_t const& src, view_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, image_t const& current, image_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, image_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, view_t const& current, image_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(image_t const& src, image_t const& current_dst)
	{
		assert(verify(src, current_dst));
		std::transform(std::execution::par, src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(image_t const& src, view_t const& current_dst)
	{
		assert(verify(src, current_dst));
		std::transform(std::execution::par, src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, image_t const& current_dst)
	{
		assert(verify(src, current_dst));
		std::transform(std::execution::par, src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, view_t const& current_dst)
	{
		assert(verify(src, current_dst));
		std::transform(std::execution::par, src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
	}

#endif // !LIBIMAGE_NO_PARALLEL
	namespace seq
	{
		void alpha_blend(image_t const& src, image_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(image_t const& src, image_t const& current_dst)
		{
			assert(verify(src, current_dst));
			std::transform(src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(image_t const& src, view_t const& current_dst)
		{
			assert(verify(src, current_dst));
			std::transform(src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, image_t const& current_dst)
		{
			assert(verify(src, current_dst));
			std::transform(src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, view_t const& current_dst)
		{
			assert(verify(src, current_dst));
			std::transform(src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
		}
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
		class PixelPlanar4
		{
		public:
			r32 red[4]   = { 0 };
			r32 green[4] = { 0 };
			r32 blue[4]  = { 0 };
			r32 alpha[4] = { 0 };
		};


		static void alpha_blend_row(pixel_t* src_begin, pixel_t* cur_begin, pixel_t* dst_begin, u32 length)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;

			r32 one = 1.0f;
			r32 u8max = 255.0f;

			auto const do_simd = [&](u32 i) 
			{
				// pixels are interleaved
				// make these 4 pixels r32 planar
				PixelPlanar4 src_mem{};
				PixelPlanar4 cur_mem{};
				PixelPlanar4 dst_mem{};

				auto src = src_begin + i;
				auto cur = cur_begin + i;
				auto dst = dst_begin + i;

				for (u32 j = 0; j < 4; ++j)
				{
					src_mem.red[j]   = (r32)src[j].red;
					src_mem.green[j] = (r32)src[j].green;
					src_mem.blue[j]  = (r32)src[j].blue;
					src_mem.alpha[j] = (r32)src[j].alpha;

					cur_mem.red[j]   = (r32)cur[j].red;
					cur_mem.green[j] = (r32)cur[j].green;
					cur_mem.blue[j]  = (r32)cur[j].blue;
				}

				auto one_vec = _mm_load_ps1(&one);
				auto u8max_vec = _mm_load_ps1(&u8max);

				auto src_a_vec = _mm_div_ps(_mm_load_ps(src_mem.alpha), u8max_vec);
				auto cur_a_vec = _mm_sub_ps(one_vec, src_a_vec);

				auto dst_vec = _mm_fmadd_ps(src_a_vec, _mm_load_ps(src_mem.red), _mm_mul_ps(cur_a_vec, _mm_load_ps(cur_mem.red)));
				_mm_store_ps(dst_mem.red, dst_vec);

				dst_vec = _mm_fmadd_ps(src_a_vec, _mm_load_ps(src_mem.green), _mm_mul_ps(cur_a_vec, _mm_load_ps(cur_mem.green)));
				_mm_store_ps(dst_mem.green, dst_vec);

				dst_vec = _mm_fmadd_ps(src_a_vec, _mm_load_ps(src_mem.blue), _mm_mul_ps(cur_a_vec, _mm_load_ps(cur_mem.blue)));
				_mm_store_ps(dst_mem.blue, dst_vec);

				for (u32 j = 0; j < 4; ++j)
				{
					dst[j].red   = (u8)dst_mem.red[j];
					dst[j].green = (u8)dst_mem.green[j];
					dst[j].blue  = (u8)dst_mem.blue[j];
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
	}

#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_COLOR