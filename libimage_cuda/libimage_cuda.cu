#include "libimage.hpp"
#include "./device/cuda_def.cuh"
#include "./device/device.hpp"

#include <algorithm>


class HSVr32
{
public:
    r32 hue;
    r32 sat;
    r32 val;
};


class RGBr32
{
public:
    r32 red;
    r32 green;
    r32 blue;
};


namespace gpuf
{    

    GPU_FUNCTION
    static HSVr32 rgb_hsv(r32 r, r32 g, r32 b)
	{
		auto max = fmaxf(r, fmaxf(g, b));
		auto min = fminf(r, fmaxf(g, b));

		auto c = max - min;

		r32 value = max;

		r32 sat = max == 0 ? 0.0f : (c / value);

		r32 hue = 60.0f;

		if (max == min)
		{
			hue = 0.0f;
		}
		else if (max == r)
		{
			hue *= ((g - b) / c);
		}
		else if (max == g)
		{
			hue *= ((b - r) / c + 2);
		}
		else // max == b
		{
			hue *= ((r - g) / c + 4);
		}

		hue /= 360.0f;

		return { hue, sat, value };
	}


    GPU_FUNCTION
    static RGBr32 hsv_rgb(r32 h, r32 s, r32 v)
	{
		auto c = s * v;
		auto m = v - c;

		auto d = h * 360.0f / 60.0f;

		auto x = c * (1.0f - fabsf(fmodf(d, 2.0f) - 1.0f));

		auto r = m;
		auto g = m;
		auto b = m;

		switch (int(d))
		{
		case 0:
			r += c;
			g += x;
			break;
		case 1:
			r += x;
			g += c;
			break;
		case 2:
			g += c;
			b += x;
			break;
		case 3:
			g += x;
			b += c;
			break;
		case 4:
			r += x;
			b += c;
			break;
		default:
			r += c;
			b += x;
			break;
		}

		return { r, g, b };
	}
}


using ViewRGBr32 = libimage::ViewRGBr32;
using ViewHSVr32 = libimage::ViewHSVr32;


GPU_KERNAL
static void gpu_map_hsv(ViewRGBr32 src, ViewHSVr32 dst, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    
}


/* map_hsv */

namespace libimage
{
	void map_hsv(ViewHSVr32 const& device_src, Image const& host_dst);

	void map_hsv(Image const& host_src, ViewHSVr32 const& device_dst);

	void map_hsv(ViewHSVr32 const& device_src, View const& host_dst);

	void map_hsv(View const& host_src, ViewHSVr32 const& device_dst);


	void map_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst);
}