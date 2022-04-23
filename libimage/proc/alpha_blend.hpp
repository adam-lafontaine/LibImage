#pragma once

#ifndef LIBIMAGE_NO_COLOR

#include "proc_def.hpp"


namespace libimage
{

	void alpha_blend(image_soa const& src, image_soa const& current, image_soa const& dst);

	void alpha_blend(image_soa const& src, image_soa const& current_dst);

#ifndef LIBIMAGE_NO_PARALLEL

	/*** alpha blend parallel ***/

	void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

	void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

	void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

	void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

	void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


	void alpha_blend(image_t const& src, image_t const& current_dst);

	void alpha_blend(image_t const& src, view_t const& current_dst);

	void alpha_blend(view_t const& src, image_t const& current_dst);

	void alpha_blend(view_t const& src, view_t const& current_dst);


#endif // !LIBIMAGE_NO_PARALLEL


	/* alpha blend seqential */

	namespace seq
	{

		void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


		void alpha_blend(image_t const& src, image_t const& current_dst);

		void alpha_blend(image_t const& src, view_t const& current_dst);

		void alpha_blend(view_t const& src, image_t const& current_dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);
	}


	/*** alpha blend simd **/

#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{

		void alpha_blend(image_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(image_t const& src, view_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, image_t const& current, view_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, image_t const& dst);

		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst);


		void alpha_blend(image_t const& src, image_t const& current_dst);

		void alpha_blend(image_t const& src, view_t const& current_dst);

		void alpha_blend(view_t const& src, image_t const& current_dst);

		void alpha_blend(view_t const& src, view_t const& current_dst);


		void alpha_blend(image_soa const& src, image_soa const& current, image_soa const& dst);

		void alpha_blend(image_soa const& src, image_soa const& current_dst);

	}

#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_COLOR