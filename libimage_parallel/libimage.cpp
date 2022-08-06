#include "libimage.hpp"


#include <execution>


#include <algorithm>
#include <cmath>


/*  libimage.cpp  */



namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void make_image(Image& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (Pixel*)malloc(sizeof(Pixel) * width * height);

		assert(image_dst.data);
	}


	void destroy_image(Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	Pixel* row_begin(Image const& image, u32 y)
	{
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	Pixel* xy_at(Image const& image, u32 x, u32 y)
	{
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}


	View make_view(Image const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		View view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	View make_view(Image& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	View sub_view(Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		View sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	View sub_view(View const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		View sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	Pixel* row_begin(View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}

	
	Pixel* xy_at(View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}


	


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void make_image(gray::Image& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (gray::Pixel*)malloc(sizeof(gray::Pixel) * width * height);

		assert(image_dst.data);
	}


	void destroy_image(gray::Image& image)
	{
		if (image.data != nullptr)
		{
			free(image.data);
			image.data = nullptr;
		}
	}


	gray::Pixel* row_begin(gray::Image const& image, u32 y)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);
		assert(y < image.height);

		auto offset = y * image.width;

		auto ptr = image.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	gray::Pixel* xy_at(gray::Image const& image, u32 x, u32 y)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);
		assert(y < image.height);
		assert(x < image.width);

		return row_begin(image, y) + x;
	}


	gray::View make_view(gray::Image const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		gray::View view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	gray::View make_view(gray::Image& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	gray::View sub_view(gray::Image const& image, Range2Du32 const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::View sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::View sub_view(gray::View const& view, Range2Du32 const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin < view.x_end);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin < view.y_end);
		assert(range.y_end <= view.y_end);

		gray::View sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::Pixel* row_begin(gray::View const& view, u32 y)
	{
		assert(y < view.height);

		auto offset = (view.y_begin + y) * view.image_width + view.x_begin;

		auto ptr = view.image_data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	gray::Pixel* xy_at(gray::View const& view, u32 x, u32 y)
	{
		assert(y < view.height);
		assert(x < view.width);

		return row_begin(view, y) + x;
	}
		

#endif // !LIBIMAGE_NO_GRAYSCALE

}


/*  verify.hpp  */

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR


	inline bool verify(View const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(Image const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(Image const& src, Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(Image const& src, View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(gray::Image const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(gray::View const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(gray::Image const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::Image const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::View const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::View const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(Image const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(Image const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, gray::Image const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(View const& src, gray::View const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
}


/*  index_range.hpp  */

#define INDEX_RANGE_IMPL
#ifdef INDEX_RANGE_IMPL

template<typename UINT>
class UnsignedRange
{
public:

	using index_type = UINT;

	class iterator
	{
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = index_type;
		using difference_type = std::ptrdiff_t;
		using pointer = const value_type*;
		using reference = value_type;

		value_type m_val;

		explicit iterator() : m_val(0) {}

		explicit iterator(value_type val) : m_val(val) {}

		reference operator*() const { return m_val; }
		iterator& operator++() { ++m_val; return *this; }
		iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
		bool operator==(iterator other) const { return m_val == other.m_val; }
		bool operator!=(iterator other) const { return !(*this == other); }
	};

private:

	index_type m_min = 0;
	index_type m_max = 1;

	template<typename INT_T>
	index_type to_min(INT_T val) { return val < 0 ? 0 : static_cast<index_type>(val); }

	template<typename INT_T>
	index_type to_max(INT_T val) { return val < 1 ? 0 : static_cast<index_type>(val - 1); }

public:

	template<typename INT_T>
	UnsignedRange(INT_T size) : m_max(to_max(size)) {}

	template<typename INT_T>
	UnsignedRange(INT_T begin, INT_T end)
	{
		if (end >= begin)
		{
			m_min = to_min(begin);
			m_max = to_max(end);
		}
		else
		{
			m_min = to_min(end);
			m_max = to_max(begin);
		}
	}

	iterator begin() { return iterator(m_min); }
	iterator end() { return iterator(m_max + 1); }
};

using u32_range_t = UnsignedRange<unsigned>;


#endif // INDEX_RANGE_IMPL


namespace libimage
{
	template <class PIXEL_T>
	static void for_each_pixel_in_row(PIXEL_T* row_begin, u32 length, std::function<void(PIXEL_T&)> const& func)
	{
		for (u32 i = 0; i < length; ++i)
		{
			func(row_begin[i]);
		}
	}


	template <class IMG_T, class PIXEL_F>
	static void for_each_pixel_by_row(IMG_T const& image, PIXEL_F const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for_each_pixel_in_row(row_begin(image, y), image.width, func);
		}
	}


	static void for_each_xy_in_row(u32 y, u32 length, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 x = 0; x < length; ++x)
		{
			func(x, y);
		}
	}


	template <class IMG_T>
	static void for_each_xy_by_row(IMG_T const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for_each_xy_in_row(y, image.width, func);
		}
	}


	template <class SRC_PIXEL_T, class DST_PIXEL_T, class SRC_TO_DST_F>
	static void transform_row(SRC_PIXEL_T* src_begin, SRC_PIXEL_T* dst_begin, u32 length, SRC_TO_DST_F const& func)
	{
		for (u32 i = 0; i < length; ++i)
		{
			dst_begin[i] = func(src_begin[i]);
		}
	}


	template <class SRC_IMG_T, class DST_IMG_T, class SRC_TO_DST_F>
	static void transform_by_row(SRC_IMG_T const& src, DST_IMG_T const& dst, SRC_TO_DST_F const& func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			transform_row(row_begin(src, y), row_begin(dst, y), src.width, func);
		}
	}


	template <class SRC_A_PIXEL_T, class SRC_B_PIXEL_T, class DST_PIXEL_T, class SRC_TO_DST_F>
	static void transform_row(SRC_A_PIXEL_T* src_a_begin, SRC_B_PIXEL_T* src_b_begin, DST_PIXEL_T* dst_begin, u32 length, SRC_TO_DST_F const& func)
	{
		for (u32 i = 0; i < length; ++i)
		{
			dst_begin[i] = func(src_a_begin[i], src_b_begin[i]);
		}
	}


	template <class SRC_A_IMG_T, class SRC_B_IMG_T, class DST_IMG_T, class SRC_TO_DST_F>
	static void transform_by_row(SRC_A_IMG_T const& src_a, SRC_B_IMG_T const& src_b, DST_IMG_T const& dst, SRC_TO_DST_F const& func)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			transform_row(row_begin(src_a, y), row_begin(src_b, y), row_begin(dst, y), src.width, func);
		}
	}


}



/* for each */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void for_each_pixel(Image const& image, std::function<void(Pixel&)> const& func)
	{
		for_each_pixel_by_row(image, func);
	}


	void for_each_pixel(View const& view, std::function<void(Pixel&)> const& func)
	{
		for_each_pixel_by_row(view, func);
	}


	void for_each_xy(Image const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for_each_xy_by_row(image, func);
	}


	void for_each_xy(View const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for_each_xy_by_row(view, func);
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void for_each_pixel(gray::Image const& image, std::function<void(gray::Pixel& p)> const& func)
	{
		for_each_pixel_by_row(image, func);
	}


	void for_each_pixel(gray::View const& view, std::function<void(gray::Pixel& p)> const& func)
	{
		for_each_pixel_by_row(view, func);
	}


	void for_each_xy(gray::Image const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for_each_xy_by_row(image, func);
	}


	void for_each_xy(gray::View const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for_each_xy_by_row(view, func);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE
}


/*  transform.cpp  */


namespace libimage
{
	



#ifndef LIBIMAGE_NO_COLOR


	void transform(Image const& src, Image const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(Image const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(View const& src, Image const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(View const& src, View const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform_in_place(Image const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p = func(p); };		
		for_each_pixel_by_row(src_dst, update);
	}


	void transform_in_place(View const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p = func(p); };
		for_each_pixel_by_row(src_dst, update);
	}


	void transform_alpha(Image const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p.alpha = func(p); };
		for_each_pixel_by_row(src_dst, update);
	}


	void transform_alpha(View const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](Pixel& p) { p.alpha = func(p); };
		for_each_pixel_by_row(src_dst, update);
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	static lookup_table_t to_lookup_table(u8_to_u8_f const& func)
	{
		lookup_table_t lut = { 0 };

		u32_range_t ids(0u, 256u);

		std::for_each(ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });

		return lut;
	}


	void transform(gray::Image const& src, gray::Image const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		transform_by_row(src, dst, conv);
	}


	void transform(gray::Image const& src, gray::View const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		transform_by_row(src, dst, conv);
	}


	void transform(gray::View const& src, gray::Image const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		transform_by_row(src, dst, conv);
	}


	void transform(gray::View const& src, gray::View const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		transform_by_row(src, dst, conv);
	}


	void transform(gray::Image const& src, gray::Image const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::Image const& src, gray::View const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::View const& src, gray::Image const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::View const& src, gray::View const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform_in_place(gray::Image const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		for_each_pixel_by_row(src_dst, conv);
	}

	void transform_in_place(gray::View const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		for_each_pixel_by_row(src_dst, conv);
	}


	void transform_in_place(gray::Image const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_in_place(src_dst, lut);
	}


	void transform_in_place(gray::View const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_in_place(src_dst, lut);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(Image const& src, gray::Image const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(Image const& src, gray::View const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(View const& src, gray::Image const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


	void transform(View const& src, gray::View const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		transform_by_row(src, dst, func);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR




}


/*  copy.cpp  */

namespace libimage
{







#ifndef LIBIMAGE_NO_COLOR

	void copy(Image const& src, Image const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(Image const& src, View const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(View const& src, Image const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(View const& src, View const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	void copy(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}


	void copy(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const func = [](auto p) { return p; };
		transform_by_row(src, dst, func);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE



}

/*  alpha_blend.cpp  */

#ifndef LIBIMAGE_NO_COLOR

namespace libimage
{
	static Pixel alpha_blend_linear(Pixel src, Pixel current)
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







	void alpha_blend(Image const& src, Image const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(Image const& src, Image const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(Image const& src, View const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(Image const& src, View const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, Image const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, Image const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, View const& current, Image const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, View const& current, View const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));
		transform_by_row(src, current, dst, alpha_blend_linear);
	}


	void alpha_blend(Image const& src, Image const& current_dst)
	{
		assert(verify(src, current_dst));
		transform_by_row(src, current_dst, current_dst, alpha_blend_linear);
	}


	void alpha_blend(Image const& src, View const& current_dst)
	{
		assert(verify(src, current_dst));
		transform_by_row(src, current_dst, current_dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, Image const& current_dst)
	{
		assert(verify(src, current_dst));
		transform_by_row(src, current_dst, current_dst, alpha_blend_linear);
	}


	void alpha_blend(View const& src, View const& current_dst)
	{
		assert(verify(src, current_dst));
		transform_by_row(src, current_dst, current_dst, alpha_blend_linear);
	}


	
}

#endif // !LIBIMAGE_NO_COLOR


/*  grayscale.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE
#ifndef LIBIMAGE_NO_COLOR

static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
	constexpr r32 COEFF_RED = 0.299f;
	constexpr r32 COEFF_GREEN = 0.587f;
	constexpr r32 COEFF_BLUE = 0.114f;

	return (u8)(COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue);
}


namespace libimage
{

	static void grayscale(u8* dst, u8* red, u8* blue, u8* green, u32 length)
	{
		for (u32 i = 0; i < length; ++i)
		{
			dst[i] = rgb_grayscale_standard(red[i], green[i], blue[i]);
		}
	}

	static constexpr u8 pixel_grayscale_standard(Pixel const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}




	void grayscale(Image const& src, gray::Image const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(Image const& src, gray::View const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(View const& src, gray::Image const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(View const& src, gray::View const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}

	void alpha_grayscale(Image const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


	void alpha_grayscale(View const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


}

#endif // !LIBIMAGE_NO_COLOR
#endif // !LIBIMAGE_NO_GRAYSCALE


/*  binary.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{


	void binarize(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize_in_place(gray::Image const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_in_place(src_dst, conv);
	}


	void binarize_in_place(gray::View const& src_dst, u8_to_bool_f const& cond)
	{
		auto const conv = [&](u8 p) { return cond(p) ? 255 : 0; };
		transform_in_place(src_dst, conv);
	}


#ifndef LIBIMAGE_NO_COLOR

	void binarize(Image const& src, gray::Image const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(Image const& src, gray::View const& dst, pixel_to_bool_f const& cond)

	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(View const& src, gray::Image const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


	void binarize(View const& src, gray::View const& dst, pixel_to_bool_f const& cond)
	{
		auto const conv = [&](Pixel p) { return cond(p) ? 255 : 0; };
		transform(src, dst, conv);
	}


#endif // !LIBIMAGE_NO_COLOR


	template <class GRAY_IMG_T>
	Point2Du32 do_centroid(GRAY_IMG_T const& src, u8_to_bool_f const& func)
	{
		constexpr u32 n_threads = 20;
		u32 h = src.height / n_threads;

		std::array<u32, n_threads> thread_totals = { 0 };
		std::array<u32, n_threads> thread_x_totals = { 0 };
		std::array<u32, n_threads> thread_y_totals = { 0 };

		u32 total = 0;
		u32 x_total = 0;
		u32 y_total = 0;

		auto const row_func = [&](u32 y)
		{
			if (y >= src.height)
			{
				return;
			}

			auto thread_id = y - n_threads * (y / n_threads);

			assert(thread_id < n_threads);

			auto row = src.row_begin(y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u32 val = func(row[x]) ? 1 : 0;

				thread_totals[thread_id] += val;
				thread_x_totals[thread_id] += x * val;
				thread_y_totals[thread_id] += y * val;
			}
		};

		for (u32 y_begin = 0; y_begin < src.height; y_begin += n_threads)
		{
			thread_totals = { 0 };
			thread_x_totals = { 0 };
			thread_y_totals = { 0 };

			u32_range_t rows(y_begin, y_begin + n_threads);

			std::for_each(rows.begin(), rows.end(), row_func);

			for (u32 i = 0; i < n_threads; ++i)
			{
				total += thread_totals[i];
				x_total += thread_x_totals[i];
				y_total += thread_y_totals[i];
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = x_total / total;
			pt.y = y_total / total;
		}

		return pt;
	}


	Point2Du32 centroid(gray::Image const& src)
	{
		assert(verify(src));

		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::Image const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::View const& src)
	{
		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::View const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}



	
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  contrast.cpp  */

constexpr u8 U8_MIN = 0;
constexpr u8 U8_MAX = 255;


#ifndef LIBIMAGE_NO_GRAYSCALE

static constexpr u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
{
	if (val < src_low)
	{
		return dst_low;
	}
	else if (val > src_high)
	{
		return dst_high;
	}

	auto const ratio = (static_cast<r64>(val) - src_low) / (src_high - src_low);

	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const diff = ratio * (dst_high - dst_low);

	return dst_low + static_cast<u8>(diff);
}


#endif // !LIBIMAGE_NO_GRAYSCALE


namespace libimage
{


#ifndef LIBIMAGE_NO_GRAYSCALE

	void contrast(gray::Image const& src, gray::Image const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::Image const& src, gray::View const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::View const& src, gray::Image const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}



	void contrast(gray::View const& src, gray::View const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast_in_place(gray::Image const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_in_place(src_dst, conv);
	}


	void contrast_in_place(gray::View const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_in_place(src_dst, conv);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE



}


/*  convolve.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
	static inline void left_2_wide(Range2Du32& range, u32 x, u32 width)
	{
		// left (2 wide)
		assert(x <= width - 2);
		range.x_begin = x;
		range.x_end = x + 2;
	}


	static inline void right_2_wide(Range2Du32& range, u32 x, u32 width)
	{
		// right (2 wide)
		assert(x >= 1);
		assert(x <= width - 1);
		range.x_begin = x - 1;
		range.x_end = x + 1;
	}


	static inline void top_2_high(Range2Du32& range, u32 y, u32 height)
	{
		// top (2 high)
		assert(y <= height - 2);
		range.y_begin = y;
		range.y_end = y + 2;
	}


	static inline void bottom_2_high(Range2Du32& range, u32 y, u32 height)
	{
		// bottom (2 high)
		assert(y >= 1);
		assert(y <= height - 1);
		range.y_begin = y - 1;
		range.y_end = y + 1;
	}


	static inline void top_or_bottom_3_high(Range2Du32& range, u32 y, u32 height)
	{
		// top or bottom (3 high)
		assert(y >= 1);
		assert(y <= height - 2);
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


	static inline void left_or_right_3_wide(Range2Du32& range, u32 x, u32 width)
	{
		// left or right (3 wide)
		assert(x >= 1);
		assert(x <= width - 2);
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


	static inline void top_or_bottom_5_high(Range2Du32& range, u32 y, u32 height)
	{
		// top or bottom (5 high)
		assert(y >= 2);
		assert(y <= height - 3);
		range.y_begin = y - 2;
		range.y_end = y + 3;
	}


	static inline void left_or_right_5_wide(Range2Du32& range, u32 x, u32 width)
	{
		// left or right (5 wide)
		assert(x >= 2);
		assert(x <= width - 3);
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


	template<class GRAY_Image, size_t N>
	static r32 apply_weights(GRAY_Image const& img, Range2Du32 const& range, std::array<r32, N> const& weights)
	{
		assert((range.y_end - range.y_begin) * (range.x_end - range.x_begin) == weights.size());

		u32 w = 0;
		r32 total = 0.0f;

		for (u32 y = range.y_begin; y < range.y_end; ++y)
		{
			auto row = img.row_begin(y);

			for (u32 x = range.x_begin; x < range.x_end; ++x)
			{
				total += weights[w++] * row[x];
			}
		}

		return total;
	}


	template<class GRAY_Image>
	static r32 weighted_center(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		Range2Du32 range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_center(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		Range2Du32 range = {};

		top_or_bottom_5_high(range, y, img.height);

		left_or_right_5_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_top_left(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		Range2Du32 range = {};

		top_2_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_top_right(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		Range2Du32 range = {};

		top_2_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_bottom_left(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		Range2Du32 range = {};

		bottom_2_high(range, y, img.width);

		left_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_bottom_right(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		Range2Du32 range = {};

		bottom_2_high(range, y, img.width);

		right_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_top(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		Range2Du32 range = {};

		top_2_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_bottom(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		Range2Du32 range = {};

		bottom_2_high(range, y, img.width);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_left(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		Range2Du32 range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	template<class GRAY_Image>
	static r32 weighted_right(GRAY_Image const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		Range2Du32 range = {};

		top_or_bottom_3_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(img, range, weights);
	}


	constexpr r32 D3 = 16.0f;
	constexpr std::array<r32, 9> GAUSS_3X3
	{
		(1 / D3), (2 / D3), (1 / D3),
		(2 / D3), (4 / D3), (2 / D3),
		(1 / D3), (2 / D3), (1 / D3),
	};

	constexpr r32 D5 = 256.0f;
	constexpr std::array<r32, 25> GAUSS_5X5
	{
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(6 / D5), (24 / D5), (36 / D5), (24 / D5), (6 / D5),
		(4 / D5), (16 / D5), (24 / D5), (16 / D5), (4 / D5),
		(1 / D5), (4 / D5),  (6 / D5),  (4 / D5),  (1 / D5),
	};


	constexpr std::array<r32, 9> GRAD_X_3X3
	{
		-0.25f,  0.0f,  0.25f,
		-0.50f,  0.0f,  0.50f,
		-0.25f,  0.0f,  0.25f,
	};


	constexpr std::array<r32, 9> GRAD_Y_3X3
	{
		-0.25f, -0.50f, -0.25f,
		 0.0f,   0.0f,   0.0f,
		 0.25f,  0.50f,  0.25f,
	};


	u8 gauss3(gray::Image const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss3(gray::View const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::Image const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::View const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	r32 x_gradient(gray::Image const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_X_3X3);
	}


	r32 x_gradient(gray::View const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_X_3X3);
	}


	r32 y_gradient(gray::Image const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_Y_3X3);
	}


	r32 y_gradient(gray::View const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_Y_3X3);
	}
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  blur.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{



	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_top(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		auto src_top = row_view(src, 0);
		auto dst_top = row_view(dst, 0);

		copy(src_top, dst_top);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		auto src_bottom = row_view(src, src.height - 1);
		auto dst_bottom = row_view(dst, src.height - 1);

		copy(src_bottom, dst_bottom);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_left(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		Range2Du32 r = {};
		r.x_begin = 0;
		r.x_end = 1;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		auto src_left = sub_view(src, r);
		auto dst_left = sub_view(dst, r);

		copy(src_left, dst_left);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void copy_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		Range2Du32 r = {};
		r.x_begin = src.width - 1;
		r.x_end = src.width;
		r.y_begin = 1;
		r.y_end = src.height - 1;

		auto src_right = sub_view(src, r);
		auto dst_right = sub_view(dst, r);

		copy(src_right, dst_right);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_top(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32 const y = 1;

		auto dst_top = dst.row_begin(y);

		u32_range_t x_ids(x_begin, x_end);

		auto const gauss = [&](u32 x)
		{
			dst_top[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_bottom(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32 const y = src.height - 2;

		auto dst_bottom = dst.row_begin(y);

		u32_range_t x_ids(x_begin, x_end);

		auto const gauss = [&](u32 x)
		{
			dst_bottom[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_left(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;
		u32 const x = 1;

		u32_range_t y_ids(y_begin, y_end);

		auto const gauss = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gauss_inner_right(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;
		u32 const x = src.width - 2;

		u32_range_t y_ids(y_begin, y_end);

		auto const gauss = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x] = gauss3(src, x, y);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 2;
		u32 const x_end = src.width - 2;
		u32_range_t x_ids(x_begin, x_end);

		auto const gauss_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const gauss_x = [&](u32 x)
			{
				dst_row[x] = gauss5(src, x, y);
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss_x);
		};

		u32 const y_begin = 2;
		u32 const y_end = src.height - 2;

		u32_range_t y_ids(y_begin, y_end);

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_blur(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		// lambdas in an array
		std::array<std::function<void()>, 9> f_list =
		{
			[&]() { copy_top(src, dst); },
			[&]() { copy_bottom(src, dst); } ,
			[&]() { copy_left(src, dst); } ,
			[&]() { copy_right(src, dst); },
			[&]() { gauss_inner_top(src, dst); },
			[&]() { gauss_inner_bottom(src, dst); },
			[&]() { gauss_inner_left(src, dst); },
			[&]() { gauss_inner_right(src, dst); },
			[&]() { inner_gauss(src, dst); }
		};

		// execute everything
		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	void blur(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}




}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  edges_gradients.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{



	template<class GRAY_IMG_T>
	static void fill_zero(GRAY_IMG_T const& view)
	{
		std::fill(std::execution::par, view.begin(), view.end(), 0);
	}


	template<class GRAY_IMG_T>
	static void zero_top(GRAY_IMG_T const& dst)
	{
		auto dst_top = row_view(dst, 0);

		fill_zero(dst_top);
	}


	template<class GRAY_IMG_T>
	static void zero_bottom(GRAY_IMG_T const& dst)
	{
		auto dst_bottom = row_view(dst, dst.height - 1);

		fill_zero(dst_bottom);
	}


	template<class GRAY_IMG_T>
	static void zero_left(GRAY_IMG_T const& dst)
	{
		Range2Du32 r = {};
		r.x_begin = 0;
		r.x_end = 1;
		r.y_begin = 1;
		r.y_end = dst.height - 1;
		auto dst_left = sub_view(dst, r);

		fill_zero(dst_left);
	}


	template<class GRAY_IMG_T>
	static void zero_right(GRAY_IMG_T const& dst)
	{
		Range2Du32 r = {};
		r.x_begin = dst.width - 1;
		r.x_end = dst.width;
		r.y_begin = 1;
		r.y_end = dst.height - 1;
		auto dst_right = sub_view(dst, r);

		fill_zero(dst_right);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = src.height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(src, x, y);
				auto gy = y_gradient(src, x, y);
				auto g = (u8)(std::hypot(gx, gy));
				dst_row[x] = cond(g) ? 255 : 0;
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
	{
		std::array<std::function<void()>, 5> f_list
		{
			[&]() { zero_top(dst); },
			[&]() { zero_bottom(dst); },
			[&]() { zero_left(dst); },
			[&]() { zero_right(dst); },
			[&]() { edges_inner(src, dst, cond); }
		};

		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		u32 const x_begin = 1;
		u32 const x_end = src.width - 1;
		u32_range_t x_ids(x_begin, x_end);

		u32 const y_begin = 1;
		u32 const y_end = src.height - 1;
		u32_range_t y_ids(y_begin, y_end);

		auto const grad_row = [&](u32 y)
		{
			auto dst_row = dst.row_begin(y);

			auto const grad_x = [&](u32 x)
			{
				auto gx = x_gradient(src, x, y);
				auto gy = y_gradient(src, x, y);
				auto g = std::hypot(gx, gy);
				dst_row[x] = (u8)(g);
			};

			std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
		};

		std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
	}


	template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
	static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
	{
		std::array<std::function<void()>, 5> f_list
		{
			[&]() { zero_top(dst); },
			[&]() { zero_bottom(dst); },
			[&]() { zero_left(dst); },
			[&]() { zero_right(dst); },
			[&]() { gradients_inner(src, dst); }
		};

		std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
	}


	void edges(gray::Image const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::Image const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::View const& src, gray::Image const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::View const& src, gray::View const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void gradients(gray::Image const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::Image const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::View const& src, gray::Image const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::View const& src, gray::View const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}




}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  rotate.cpp  */

static Point2Dr32 find_rotation_src(Point2Du32 const& pt, Point2Du32 const& origin, r32 theta_rotate)
{
	auto dx_dst = (r32)pt.x - (r32)origin.x;
	auto dy_dst = (r32)pt.y - (r32)origin.y;

	auto radius = std::hypotf(dx_dst, dy_dst);

	auto theta_dst = atan2f(dy_dst, dx_dst);
	auto theta_src = theta_dst - theta_rotate;

	auto dx_src = radius * cosf(theta_src);
	auto dy_src = radius * sinf(theta_src);

	Point2Dr32 pt_src{};
	pt_src.x = (r32)origin.x + dx_src;
	pt_src.y = (r32)origin.y + dy_src;

	return pt_src;
}


namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	template <typename IMG_T>
	static Pixel get_color(IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return to_pixel(0, 0, 0);
		}

		return *src_image.xy_at((u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_T>
	static u8 get_gray(GR_IMG_T const& src_image, Point2Dr32 location)
	{
		auto zero = 0.0f;
		auto width = (r32)src_image.width;
		auto height = (r32)src_image.height;

		auto x = location.x;
		auto y = location.y;

		if (x < zero || x >= width || y < zero || y >= height)
		{
			return 0;
		}

		return *src_image.xy_at((u32)floorf(x), (u32)floorf(y));
	}

#endif // !LIBIMAGE_NO_GRAYSCALE




#ifndef LIBIMAGE_NO_COLOR	


	template <typename IMG_SRC_T, typename IMG_DST_T>
	static void rotate_par(IMG_SRC_T const& src, IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		u32_range_t range(dst.width * dst.height);

		auto const func = [&](u32 i)
		{
			auto y = i / dst.width;
			auto x = i - y * dst.width;
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*dst.xy_at(x, y) = get_color(src, src_pt);
		};

		std::for_each(std::execution::par, range.begin(), range.end(), func);
	}


	void rotate(Image const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(Image const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(View const& src, Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(View const& src, View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(Image const& src, Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(Image const& src, View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(View const& src, Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(View const& src, View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	template <typename GR_IMG_SRC_T, typename GR_IMG_DST_T>
	static void rotate_par_gray(GR_IMG_SRC_T const& src, GR_IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		Point2Du32 origin = { origin_x, origin_y };

		u32_range_t range(dst.width * dst.height);

		auto const func = [&](u32 i)
		{
			auto y = i / dst.width;
			auto x = i - y * dst.width;
			auto src_pt = find_rotation_src({ x, y }, origin, theta);
			*dst.xy_at(x, y) = get_gray(src, src_pt);
		};

		std::for_each(std::execution::par, range.begin(), range.end(), func);
	}


	void rotate(gray::Image const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::Image const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::View const& src, gray::Image const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::View const& src, gray::View const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::Image const& src, gray::Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::Image const& src, gray::View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::View const& src, gray::Image const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::View const& src, gray::View const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


}