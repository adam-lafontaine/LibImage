#include "process.hpp"
#include "verify.hpp"

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_SIMD
#include <xmmintrin.h>
#include <immintrin.h>
#endif // !LIBIMAGE_NO_SIMD


static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
	return static_cast<u8>(0.299 * red + 0.587 * green + 0.114 * blue);
}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	static constexpr u8 pixel_grayscale_standard(pixel_t const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


	void grayscale(image_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(image_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(view_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(view_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_COLOR

	void alpha_grayscale(image_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


	void alpha_grayscale(view_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}

#endif // !LIBIMAGE_NO_COLOR




#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void grayscale(image_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(image_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(view_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(view_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_COLOR

		void alpha_grayscale(image_t const& src)
		{
			seq::transform_alpha(src, pixel_grayscale_standard);
		}


		void alpha_grayscale(view_t const& src)
		{
			seq::transform_alpha(src, pixel_grayscale_standard);
		}

#endif // !LIBIMAGE_NO_COLOR

	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


		class PixelPlanar4
		{
		public:
			r32 red[4] = { 0 };
			r32 green[4] = { 0 };
			r32 blue[4] = { 0 };
			r32 alpha[4] = { 0 };
		};


		static void copy_4(pixel_t* src, PixelPlanar4& dst)
		{
			for (u32 i = 0; i < 4; ++i)
			{
				dst.red[i] = src[i].red;
				dst.green[i] = src[i].green;
				dst.blue[i] = src[i].blue;
				dst.alpha[i] = src[i].alpha;
			}
		}


		template <typename SRC_T, typename DST_T>
		static void copy_4(SRC_T* src, DST_T* dst)
		{
			dst[0] = (DST_T)src[0];
			dst[1] = (DST_T)src[1];
			dst[2] = (DST_T)src[2];
			dst[3] = (DST_T)src[3];
		}


		static void grayscale_row(pixel_t* src_begin, u8* dst_begin, u32 length)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;

			r32 weights[] = { 0.299f, 0.587f, 0.114f };
			auto red_w_vec = _mm_load_ps1(weights);
			auto green_w_vec = _mm_load_ps1(weights + 1);
			auto blue_w_vec = _mm_load_ps1(weights + 2);

			auto const do_simd = [&](u32 i)
			{
				// pixels are interleaved
				// make these 4 pixels r32 planar
				PixelPlanar4 mem{};
				copy_4(src_begin + i, mem);

				auto src_vec = _mm_load_ps(mem.red);
				auto dst_vec = _mm_mul_ps(src_vec, red_w_vec);

				src_vec = _mm_load_ps(mem.green);
				dst_vec = _mm_fmadd_ps(src_vec, green_w_vec, dst_vec);

				src_vec = _mm_load_ps(mem.blue);
				dst_vec = _mm_fmadd_ps(src_vec, blue_w_vec, dst_vec);

				_mm_store_ps(mem.red, dst_vec);

				copy_4(mem.red, dst_begin + i);
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		void grayscale(image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			grayscale_row(src.begin(), dst.begin(), src.width * src.height);
		}


		void grayscale(image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				grayscale_row(src.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void grayscale(view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				grayscale_row(src.row_begin(y), dst.row_begin(y), src.width);
			}
		}


		void grayscale(view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			for (u32 y = 0; y < src.height; ++y)
			{
				grayscale_row(src.row_begin(y), dst.row_begin(y), src.width);
			}
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_COLOR




		static void alpha_grayscale_row(pixel_t* src_begin, u32 length)
		{
			constexpr u32 N = 4;
			constexpr u32 STEP = N;

			r32 weights[] = { 0.299f, 0.587f, 0.114f };
			auto red_w_vec = _mm_load_ps1(weights);
			auto green_w_vec = _mm_load_ps1(weights + 1);
			auto blue_w_vec = _mm_load_ps1(weights + 2);

			auto const do_simd = [&](u32 i)
			{
				// pixels are interleaved
				// make these 4 pixels r32 planar
				PixelPlanar4 mem{};
				copy_4(src_begin + i, mem);

				auto src_vec = _mm_load_ps(mem.red);
				auto dst_vec = _mm_mul_ps(src_vec, red_w_vec);

				src_vec = _mm_load_ps(mem.green);
				dst_vec = _mm_fmadd_ps(src_vec, green_w_vec, dst_vec);

				src_vec = _mm_load_ps(mem.blue);
				dst_vec = _mm_fmadd_ps(src_vec, blue_w_vec, dst_vec);

				_mm_store_ps(mem.alpha, dst_vec);

				for (u32 j = 0; j < 4; ++j)
				{
					(src_begin + i)[j].alpha = (u8)mem.alpha[j];
				}
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		void alpha_grayscale(image_t const& src)
		{
			assert(verify(src));

			alpha_grayscale_row(src.begin(), src.width * src.height);
		}


		void alpha_grayscale(view_t const& src)
		{
			assert(verify(src));

			for (u32 y = 0; y < src.height; ++y)
			{
				alpha_grayscale_row(src.row_begin(y), src.width);
			}
		}

#endif // !LIBIMAGE_NO_COLOR

	}

#endif // !LIBIMAGE_NO_SIMD
}