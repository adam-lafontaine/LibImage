#pragma once

#include "defines.hpp"


namespace libimage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;


	class PlatformPixel
	{
	public:

		u8 red = 0;
		u8 green = 0;
		u8 blue = 0;
		u8 alpha = 0;
	};


	class PlatformImage
	{
	public:

		u32 width = 0;
		u32 height = 0;

		PlatformPixel* data = nullptr;
	};


	void make_image(PlatformImage& image, u32 width, u32 height);

	void destroy_image(PlatformImage& image);


	class ImageRGBAr32
	{
	public:

		u32 width = 0;
		u32 height = 0;

		r32* red = nullptr;
		r32* green = nullptr;
		r32* blue = nullptr;
		r32* alpha = nullptr;
	};


	void make_image(ImageRGBAr32& image, u32 width, u32 height);

	void destroy_image(ImageRGBAr32& image);



}


