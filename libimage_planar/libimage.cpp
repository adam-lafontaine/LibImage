#include "libimage.hpp"

#include <cstdlib>

namespace libimage
{
	void make_image(PlatformImage& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);
		
		image.data = (PlatformPixel*)malloc(sizeof(PlatformPixel) * width * height);
		assert(image.data);

		image.width = width;
		image.height = height;
	}


	void destroy_image(PlatformImage& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	void make_image(ImageRGBAr32& image, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		auto n_pixels = width * height;

		auto data = (r32*)malloc(sizeof(r32) * 4 * n_pixels);
		assert(data);

		image.width = width;
		image.height = height;

		image.red = data;
		image.green = image.red + n_pixels;
		image.blue = image.green + n_pixels;
		image.alpha = image.blue + n_pixels;
	}


	void destroy_image(ImageRGBAr32& image)
	{
		if (image.red != nullptr)
		{
			free(image.red);
			image.red = nullptr;
			image.green = nullptr;
			image.blue = nullptr;
			image.alpha = nullptr;
		}
	}
}


