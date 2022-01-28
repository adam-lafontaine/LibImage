#ifndef LIBIMAGE_NO_SIMD

#include "../verify.hpp"
#include "simd_def.hpp"


#include <array>


constexpr r32 COEFF_RED = 0.299f;
constexpr r32 COEFF_GREEN = 0.587f;
constexpr r32 COEFF_BLUE = 0.114f;

constexpr std::array<r32, 3> STANDARD_GRAYSCALE_COEFFS{ COEFF_RED, COEFF_GREEN, COEFF_BLUE };


namespace libimage
{
	namespace simd
	{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE	


		static void grayscale_row(pixel_t* src_begin, u8* dst_begin, u32 length)
		{
			constexpr u32 STEP = VEC_LEN;

			auto weights = STANDARD_GRAYSCALE_COEFFS.data();

			auto red_w_vec = simd_load_broadcast(weights);
			auto green_w_vec = simd_load_broadcast(weights + 1);
			auto blue_w_vec = simd_load_broadcast(weights + 2);

			auto const do_simd = [&](u32 i)
			{
				// pixels are interleaved
				// make them planar
				PixelPlanar mem{};
				copy_vec_len(src_begin + i, mem);

				auto src_vec = simd_load(mem.red);
				auto dst_vec = simd_multiply(src_vec, red_w_vec);

				src_vec = simd_load(mem.green);
				dst_vec = simd_fmadd(src_vec, green_w_vec, dst_vec);

				src_vec = simd_load(mem.blue);
				dst_vec = simd_fmadd(src_vec, blue_w_vec, dst_vec);

				simd_store(mem.alpha, dst_vec);

				cast_copy_len(mem.alpha, dst_begin + i);
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


		static void grayscale_row(u8* red, u8* green, u8* blue, u8* dst, u32 length)
		{
			constexpr u32 STEP = VEC_LEN;

			auto weights = STANDARD_GRAYSCALE_COEFFS.data();

			auto red_w_vec = simd_load_broadcast(weights);
			auto green_w_vec = simd_load_broadcast(weights + 1);
			auto blue_w_vec = simd_load_broadcast(weights + 2);

			auto const do_simd = [&](u32 i)
			{
				PixelPlanar mem{};
				copy_vec_len(red + i, green + i, blue + i, mem);

				auto src_vec = simd_load(mem.red);
				auto dst_vec = simd_multiply(src_vec, red_w_vec);

				src_vec = simd_load(mem.green);
				dst_vec = simd_fmadd(src_vec, green_w_vec, dst_vec);

				src_vec = simd_load(mem.blue);
				dst_vec = simd_fmadd(src_vec, blue_w_vec, dst_vec);

				simd_store(mem.alpha, dst_vec);

				cast_copy_len(mem.alpha, dst + i);
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		void grayscale(image_soa const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			grayscale_row(src.red, src.green, src.blue, dst.data, src.width * src.height);
		}


		void grayscale(image_soa const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			auto red = src.red;
			auto green = src.green;
			auto blue = src.blue;

			for (u32 y = 0; y < src.height; ++y)
			{
				grayscale_row(red, green, blue, dst.row_begin(y), src.width);
				
				red += src.width;
				green += src.width;
				blue += src.width;
			}
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_COLOR


		static void alpha_grayscale_row(pixel_t* src_begin, u32 length)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = VEC_LEN;

			auto weights = STANDARD_GRAYSCALE_COEFFS.data();

			auto red_w_vec = simd_load_broadcast(weights);
			auto green_w_vec = simd_load_broadcast(weights + 1);
			auto blue_w_vec = simd_load_broadcast(weights + 2);

			auto const do_simd = [&](u32 i)
			{
				// pixels are interleaved
				// make these 4 pixels r32 planar
				PixelPlanar mem{};
				copy_vec_len(src_begin + i, mem);

				auto src_vec = simd_load(mem.red);
				auto dst_vec = simd_multiply(src_vec, red_w_vec);

				src_vec = simd_load(mem.green);
				dst_vec = simd_fmadd(src_vec, green_w_vec, dst_vec);

				src_vec = simd_load(mem.blue);
				dst_vec = simd_fmadd(src_vec, blue_w_vec, dst_vec);

				simd_store(mem.alpha, dst_vec);

				for (u32 j = 0; j < N; ++j)
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


}

#endif // !LIBIMAGE_NO_SIMD