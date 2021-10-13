/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_COLOR

#include "process.hpp"
#include "verify.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{
	static pixel_t alpha_blend_linear(pixel_t const& src, pixel_t const& current)
	{
		auto const a = static_cast<r32>(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = static_cast<r32>(s);
			auto cf = static_cast<r32>(c);

			auto blended = a * cf + (1.0f - a) * sf;

			return static_cast<u8>(blended);
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