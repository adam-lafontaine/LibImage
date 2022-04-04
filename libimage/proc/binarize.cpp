#include "process.hpp"
#include "verify.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}


	void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}


#ifndef LIBIMAGE_NO_COLOR

	void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)

	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


#endif // !LIBIMAGE_NO_COLOR

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform_self(src_dst, conv);
		}


		void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& cond)
		{
			auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
			seq::transform_self(src_dst, conv);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
	}

}