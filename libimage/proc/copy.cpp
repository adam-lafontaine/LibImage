#include "process.hpp"
#include "verify.hpp"

#include <algorithm>

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void copy(image_soa const& src, image_t const& dst)
	{
		assert(verify(src, dst));

		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = dst.data[i];
			p.red = src.red[i];
			p.green = src.green[i];
			p.blue = src.blue[i];
			p.alpha = src.alpha[i];
		}
	}


	void copy(image_t const& src, image_soa const& dst)
	{
		assert(verify(src, dst));

		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = src.data[i];
			dst.red[i] = p.red;
			dst.green[i] = p.green;
			dst.blue[i] = p.blue;
			dst.alpha[i] = p.alpha;
		}
	}


	void copy(image_soa const& src, view_t const& dst)
	{
		auto dst_it = dst.begin();
		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = *dst_it;
			p.red = src.red[i];
			p.green = src.green[i];
			p.blue = src.blue[i];
			p.alpha = src.alpha[i];

			++dst_it;
		}
	}


	void copy(view_t const& src, image_soa const& dst)
	{
		auto src_it = src.begin();
		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = *src_it;
			dst.red[i] = p.red;
			dst.green[i] = p.green;
			dst.blue[i] = p.blue;
			dst.alpha[i] = p.alpha;

			++src_it;
		}
	}


	void copy(image_soa const& src, image_soa const& dst)
	{
		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			dst.red[i] = src.red[i];
			dst.green[i] = src.green[i];
			dst.blue[i] = src.blue[i];
			dst.alpha[i] = src.alpha[i];
		}
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_PARALLEL


#ifndef LIBIMAGE_NO_COLOR

	void copy(image_t const& src, image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(image_t const& src, view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(view_t const& src, image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(view_t const& src, view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	void copy(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}


	void copy(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL
	
	
	
	
	namespace seq
	{

#ifndef LIBIMAGE_NO_COLOR

		void copy(image_t const& src, image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(image_t const& src, view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(view_t const& src, image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(view_t const& src, view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE


		void copy(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


		void copy(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			std::copy(src.begin(), src.end(), dst.begin());
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}


}

