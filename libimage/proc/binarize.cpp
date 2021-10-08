/*

Copyright (c) 2021 Adam Lafontaine

*/

#ifndef LIBIMAGE_NO_GRAYSCALE

#include "process.hpp"
#include "verify.hpp"

#include <algorithm>
#include <execution>

namespace libimage
{
	void binarize(gray::image_t const& src, gray::image_t const& dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::image_t const& src, gray::view_t const& dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::image_t const& dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_self(gray::image_t const& src_dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform_self(src_dst, conv);
	}


	void binarize_self(gray::view_t const& src_dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		transform_self(src_dst, conv);
	}


	void binarize(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_self(gray::image_t const& src_dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}


	void binarize_self(gray::view_t const& src_dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		transform_self(src_dst, conv);
	}



	namespace seq
	{
		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold)
		{
			auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, u8 min_threashold)
		{
			auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
			seq::transform(src, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
		{
			auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(gray::view_t const& src, u8_to_bool_f const& func)
		{
			auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
			seq::transform(src, conv);
		}
	}
}

#endif // !LIBIMAGE_NO_GRAYSCALE