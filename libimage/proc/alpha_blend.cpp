#ifndef LIBIMAGE_NO_COLOR

#include "process.hpp"
#include "verify.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{
	static u8 alpha_blend_linear_soa(u8 src, u8 current, u8 alpha)
	{
		auto const a = alpha / 255.0f;

		auto sf = (r32)(src);
		auto cf = (r32)(current);

		auto blended = a * sf + (1.0f - a) * cf;

		return (u8)(blended);
	}



	void alpha_blend(image_soa const& src, image_soa const& current, image_soa const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		for (u32 i = 0; i < src.width * src.height; ++i)
		{			
			dst.red[i] = alpha_blend_linear_soa(src.red[i], current.red[i], src.alpha[i]);
			dst.green[i] = alpha_blend_linear_soa(src.green[i], current.green[i], src.alpha[i]);
			dst.blue[i] = alpha_blend_linear_soa(src.blue[i], current.blue[i], src.alpha[i]);
			dst.alpha[i] = 255;
		}
	}


	void alpha_blend(image_soa const& src, image_soa const& current_dst)
	{
		alpha_blend(src, current_dst, current_dst);
	}


	static pixel_t alpha_blend_linear(pixel_t const& src, pixel_t const& current)
	{
		auto const a = (r32)(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = (r32)(s);
			auto cf = (r32)(c);

			auto blended = a * sf + (1.0f - a) * cf;

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


}

#endif // !LIBIMAGE_NO_COLOR