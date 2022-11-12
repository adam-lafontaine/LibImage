#include "libimage.hpp"

/* verify */

#ifndef NDEBUG

namespace libimage
{
	static bool verify(Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(View const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	template <size_t N>
	static bool verify(ViewCHr32<N> const& view)
	{
		return view.image_width && view.width && view.height && view.image_channel_data[0];
	}


	static bool verify(gray::Image const& image)
	{
		return image.width && image.height && image.data;
	}


	static bool verify(gray::View const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	static bool verify(View1r32 const& view)
	{
		return view.image_width && view.width && view.height && view.image_data;
	}


	template <class IMG_A, class IMG_B>
	static bool verify(IMG_A const& lhs, IMG_B const& rhs)
	{
		return
			verify(lhs) && verify(rhs) &&
			lhs.width == rhs.width &&
			lhs.height == rhs.height;
	}


	template <class IMG>
	static bool verify(IMG const& image, Range2Du32 const& range)
	{
		return
			verify(image) &&
			range.x_begin < range.x_end &&
			range.y_begin < range.y_end &&
			range.x_begin < image.width &&
			range.x_end <= image.width &&
			range.y_begin < image.height &&
			range.y_end <= image.height;
	}
}

#endif // !NDEBUG


namespace libimage
{
	/*
	constexpr inline std::array<r32, 256> channel_r32_lut()
	{
		std::array<r32, 256> lut = {};

		for (u32 i = 0; i < 256; ++i)
		{
			lut[i] = i / 255.0f;
		}

		return lut;
	}*/


	constexpr inline r32 to_channel_r32(u8 value)
	{
		//constexpr auto lut = channel_r32_lut();

		//return lut[value];

		return value / 255.0f;
	}


	constexpr inline u8 to_channel_u8(r32 value)
	{
		if (value < 0.0f)
		{
			value = 0.0f;
		}
		else if (value > 1.0f)
		{
			value = 1.0f;
		}

		return (u8)(u32)(value * 255 + 0.5f);
	}
}