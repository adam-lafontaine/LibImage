#pragma once

#include "proc_def.hpp"


namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE


	void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);


	void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

	void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


	namespace fast
	{
		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);


		namespace fast
		{
			void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond);


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);


		namespace fast
		{
			void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond);

			void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
	}

#endif // !LIBIMAGE_NO_SIMD
}