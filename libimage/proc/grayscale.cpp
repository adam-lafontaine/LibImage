#include "process.hpp"
#include "verify.hpp"

#include <array>
#include <functional>
#include <execution>


constexpr r32 COEFF_RED = 0.299f;
constexpr r32 COEFF_GREEN = 0.587f;
constexpr r32 COEFF_BLUE = 0.114f;


static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
	return static_cast<u8>(COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue);
}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


	static void grayscale(u8* dst, u8* red, u8* blue, u8* green, u32 length)
	{
		for (u32 i = 0; i < length; ++i)
		{
			dst[i] = rgb_grayscale_standard(red[i], green[i], blue[i]);
		}
	}


	void grayscale(image_soa const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			dst.data[i] = rgb_grayscale_standard(src.red[i], src.green[i], src.blue[i]);
		}
	}


	void grayscale(image_soa const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		auto dst_it = dst.begin();
		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = *dst_it;
			p = rgb_grayscale_standard(src.red[i], src.green[i], src.blue[i]);

			++dst_it;
		}
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR



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
}