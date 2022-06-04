#include "libimage.hpp"

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

#include <algorithm>
#include <cmath>


/*  libimage.cpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void make_image(image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (pixel_t*)malloc(sizeof(pixel_t) * width * height);

		assert(image_dst.data);
	}


	view_t make_view(image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		view_t view;

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


	view_t make_view(image_t& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	view_t sub_view(image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		view_t sub_view;

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


	view_t sub_view(view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		view_t sub_view;

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


	view_t row_view(image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}


	void for_each_pixel(image_t const& image, std::function<void(pixel_t& p)> const& func)
	{
		u32 size = image.width * image.height;
		for (u32 i = 0; i < size; ++i)
		{
			func(image.data[i]);
		}
	}


	void for_each_pixel(view_t const& view, std::function<void(pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto row = view.row_begin(y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_xy(image_t const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for (u32 x = 0; x < image.width; ++x)
			{
				func(x, y);
			}
		}
	}


	void for_each_xy(view_t const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 x = 0; x < view.width; ++x)
			{
				func(x, y);
			}
		}
	}


	void make_planar(image_soa& dst, u32 width, u32 height)
	{
		assert(!dst.data);

		size_t image_sz = (size_t)width * height;

		dst.data = (u8*)malloc(4 * image_sz);

		if (!dst.data)
		{
			return;
		}

		dst.width = width;
		dst.height = height;

		dst.red = dst.data;
		dst.green = dst.red + image_sz;
		dst.blue = dst.green + image_sz;
		dst.alpha = dst.blue + image_sz;
	}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void make_image(gray::image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (gray::pixel_t*)malloc(sizeof(gray::pixel_t) * width * height);

		assert(image_dst.data);
	}


	gray::view_t make_view(gray::image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		gray::view_t view;

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


	gray::view_t make_view(gray::image_t& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::view_t sub_view;

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


	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin < view.x_end);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin < view.y_end);
		assert(range.y_end <= view.y_end);

		gray::view_t sub_view;

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


	gray::view_t row_view(gray::image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}


	void for_each_pixel(gray::image_t const& image, std::function<void(gray::pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			auto row = image.row_begin(y);
			for (u32 x = 0; x < image.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_pixel(gray::view_t const& view, std::function<void(gray::pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto row = view.row_begin(y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_xy(gray::image_t const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for (u32 x = 0; x < image.width; ++x)
			{
				func(x, y);
			}
		}
	}


	void for_each_xy(gray::view_t const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 x = 0; x < view.width; ++x)
			{
				func(x, y);
			}
		}
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

}


/*  verify.hpp  */

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR


	inline bool verify(view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(image_t const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(image_soa const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(image_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_soa const& src, image_soa const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_t const& src, image_soa const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_soa const& src, image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, image_soa const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_soa const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(gray::image_t const& img)
	{
		return img.data && img.width && img.height;
	}


	inline bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	inline bool verify(gray::image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_soa const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_soa const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	inline bool verify(image_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(image_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, gray::image_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	inline bool verify(view_t const& src, gray::view_t const& dst)
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


/*  copy.cpp  */

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



/*  transform.cpp  */

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR


	void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


	void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const update = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), update);
	}


#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func)
	{
		lookup_table_t lut = { 0 };

		u32_range_t ids(0u, 256u);

		std::for_each(std::execution::par, ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });

		return lut;
	}


	void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
	{
		assert(verify(src, dst));
		auto const conv = [&lut](u8 p) { return lut[p]; };
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), conv);
	}


	void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));
		auto const lut = to_lookup_table(func);
		transform(src, dst, lut);
	}


	void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), conv);
	}

	void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut)
	{
		assert(verify(src_dst));
		auto const conv = [&lut](u8& p) { p = lut[p]; };
		std::for_each(std::execution::par, src_dst.begin(), src_dst.end(), conv);
	}


	void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_self(src_dst, lut);
	}


	void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func)
	{
		assert(verify(src_dst));
		auto const lut = to_lookup_table(func);
		transform_self(src_dst, lut);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


	void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));
		std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR


#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{

#ifndef LIBIMAGE_NO_COLOR


		void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src_dst));
			auto const update = [&](pixel_t& p) { p = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), update);
		}


		void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func)
		{
			assert(verify(src_dst));
			auto const update = [&](pixel_t& p) { p = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), update);
		}


		void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const conv = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}


		void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const conv = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		lookup_table_t to_lookup_table(u8_to_u8_f const& func)
		{
			lookup_table_t lut = { 0 };

			u32_range_t ids(0u, 256u);

			std::for_each(ids.begin(), ids.end(), [&](u32 id) { lut[id] = func(id); });

			return lut;
		}


		void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut)
		{
			assert(verify(src, dst));
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));
			auto const lut = seq::to_lookup_table(func);
			auto const conv = [&lut](u8 p) { return lut[p]; };
			std::transform(src.begin(), src.end(), dst.begin(), conv);
		}


		void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut)
		{
			assert(verify(src_dst));
			auto const conv = [&lut](u8& p) { p = lut[p]; };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}

		void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut)
		{
			assert(verify(src_dst));
			auto const conv = [&lut](u8& p) { p = lut[p]; };
			std::for_each(src_dst.begin(), src_dst.end(), conv);
		}


		void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const lut = to_lookup_table(func);
			seq::transform_self(src_dst, lut);
		}


		void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func)
		{
			assert(verify(src_dst));
			auto const lut = to_lookup_table(func);
			seq::transform_self(src_dst, lut);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE


#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE


		void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}


		void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));
			std::transform(src.begin(), src.end(), dst.begin(), func);
		}

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
	}

}


/*  alpha_blend.cpp  */

#ifndef LIBIMAGE_NO_COLOR

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


/*  grayscale.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE
#ifndef LIBIMAGE_NO_COLOR

static constexpr u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
	constexpr r32 COEFF_RED = 0.299f;
	constexpr r32 COEFF_GREEN = 0.587f;
	constexpr r32 COEFF_BLUE = 0.114f;

	return static_cast<u8>(COEFF_RED * red + COEFF_GREEN * green + COEFF_BLUE * blue);
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


	void grayscale(image_soa const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			dst.data[i] = rgb_grayscale_standard(src.red[i], src.green[i], src.blue[i]);
		}
	}


	void grayscale(image_soa const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		auto dst_it = dst.begin();
		for (u32 i = 0; i < src.width * src.height; ++i)
		{
			auto& p = *dst_it;
			p = rgb_grayscale_standard(src.red[i], src.green[i], src.blue[i]);

			++dst_it;
		}
	}

	static constexpr u8 pixel_grayscale_standard(pixel_t const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}

#ifndef LIBIMAGE_NO_PARALLEL


	void grayscale(image_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(image_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(view_t const& src, gray::image_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}


	void grayscale(view_t const& src, gray::view_t const& dst)
	{
		transform(src, dst, pixel_grayscale_standard);
	}

	void alpha_grayscale(image_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}


	void alpha_grayscale(view_t const& src)
	{
		transform_alpha(src, pixel_grayscale_standard);
	}

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{

		void grayscale(image_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(image_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(view_t const& src, gray::image_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}


		void grayscale(view_t const& src, gray::view_t const& dst)
		{
			seq::transform(src, dst, pixel_grayscale_standard);
		}

		void alpha_grayscale(image_t const& src)
		{
			seq::transform_alpha(src, pixel_grayscale_standard);
		}


		void alpha_grayscale(view_t const& src)
		{
			seq::transform_alpha(src, pixel_grayscale_standard);
		}

	}
}

#endif // !LIBIMAGE_NO_COLOR
#endif // !LIBIMAGE_NO_GRAYSCALE


/*  binary.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL

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

			std::for_each(std::execution::par, rows.begin(), rows.end(), row_func);

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


	Point2Du32 centroid(gray::image_t const& src)
	{
		assert(verify(src));

		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::view_t const& src)
	{
		auto const func = [](u8 p) { return p > 0; };
		return do_centroid(src, func);
	}


	Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func)
	{
		assert(verify(src));

		return do_centroid(src, func);
	}

#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
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


#ifndef LIBIMAGE_NO_COLOR

		void binarize(image_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(image_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(view_t const& src, gray::image_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


		void binarize(view_t const& src, gray::view_t const& dst, pixel_to_bool_f const& cond)
		{
			auto const conv = [&](pixel_t p) { return cond(p) ? 255 : 0; };
			seq::transform(src, dst, conv);
		}


#endif // !LIBIMAGE_NO_COLOR


		template <class GRAY_IMG_T>
		Point2Du32 do_centroid(GRAY_IMG_T const& src, u8_to_bool_f const& func)
		{
			u32 total = 0;
			u32 x_total = 0;
			u32 y_total = 0;

			auto const xy_func = [&](u32 x, u32 y)
			{
				u32 val = func(*src.xy_at(x, y)) ? 1 : 0;

				total += val;
				x_total += x * val;
				y_total += y * val;
			};

			for_each_xy(src, xy_func);

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


		Point2Du32 centroid(gray::image_t const& src)
		{
			assert(verify(src));

			auto const func = [](u8 p) { return p > 0; };
			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::image_t const& src, u8_to_bool_f const& func)
		{
			assert(verify(src));

			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::view_t const& src)
		{
			assert(verify(src));

			auto const func = [](u8 p) { return p > 0; };
			return seq::do_centroid(src, func);
		}


		Point2Du32 centroid(gray::view_t const& src, u8_to_bool_f const& func)
		{
			assert(verify(src));

			return seq::do_centroid(src, func);
		}


		template <class GRAY_IMG_T>
		static bool do_neighbors(GRAY_IMG_T const& img, u32 x, u32 y)
		{
			assert(x >= 1);
			assert(x < img.width);
			assert(y >= 1);
			assert(y < img.height);

			constexpr std::array<int, 8> x_neighbors = { -1,  0,  1,  1,  1,  0, -1, -1 };
			constexpr std::array<int, 8> y_neighbors = { -1, -1, -1,  0,  1,  1,  1,  0 };

			constexpr auto n_neighbors = x_neighbors.size();
			int value_total = 0;
			u32 value_count = 0;
			u32 flip = 0;

			auto xi = (u32)(x + x_neighbors[n_neighbors - 1]);
			auto yi = (u32)(y + y_neighbors[n_neighbors - 1]);
			auto val = *img.xy_at(xi, yi);
			bool is_on = val != 0;

			for (u32 i = 0; i < n_neighbors; ++i)
			{
				xi = (u32)(x + x_neighbors[i]);
				yi = (u32)(y + y_neighbors[i]);

				val = *img.xy_at(xi, yi);
				flip += (val != 0) != is_on;

				is_on = val != 0;
				value_count += is_on;
			}

			return value_count > 1 && value_count < 7 && flip == 2;
		}


		template <class GRAY_IMG_T>
		static u32 thin_once(GRAY_IMG_T const& img)
		{
			u32 pixel_count = 0;

			auto width = img.width;
			auto height = img.height;

			auto const xy_func = [&](u32 x, u32 y)
			{
				auto& p = *img.xy_at(x, y);
				if (p == 0)
				{
					return;
				}

				if (do_neighbors(img, x, y))
				{
					p = 0;
				}

				pixel_count += p > 0;
			};

			u32 x_begin = 1;
			u32 x_end = width - 1;
			u32 y_begin = 1;
			u32 y_end = height - 2;
			u32 x = 0;
			u32 y = 0;

			auto const done = [&]() { return !(x_begin < x_end&& y_begin < y_end); };

			while (!done())
			{
				// iterate clockwise
				y = y_begin;
				x = x_begin;
				for (; x < x_end; ++x)
				{
					xy_func(x, y);
				}
				--x;

				for (++y; y < y_end; ++y)
				{
					xy_func(x, y);
				}
				--y;

				for (--x; x >= x_begin; --x)
				{
					xy_func(x, y);
				}
				++x;

				for (--y; y > y_begin; --y)
				{
					xy_func(x, y);
				}
				++y;

				++x_begin;
				++y_begin;
				--x_end;
				--y_end;

				if (done())
				{
					break;
				}

				// iterate counter clockwise
				for (++x; y < y_end; ++y)
				{
					xy_func(x, y);
				}
				--y;

				for (++x; x < x_end; ++x)
				{
					xy_func(x, y);
				}
				--x;

				for (--y; y >= y_begin; --y)
				{
					xy_func(x, y);
				}
				++y;

				for (--x; x >= x_begin; --x)
				{
					xy_func(x, y);
				}
				++x;

				++x_begin;
				++y_begin;
				--x_end;
				--y_end;
			}

			return pixel_count;
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_thin(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::copy(src, dst);

			u32 current_count = 0;
			u32 pixel_count = thin_once(dst);
			u32 max_iter = 100; // src.width / 2;

			for (u32 i = 1; pixel_count != current_count && i < max_iter; ++i)
			{
				current_count = pixel_count;
				pixel_count = thin_once(dst);
			}
		}


		void thin_objects(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			do_thin(src, dst);
		}


		void thin_objects(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			do_thin(src, dst);
		}


		void thin_objects(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			do_thin(src, dst);
		}


		void thin_objects(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			do_thin(src, dst);
		}

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
#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_GRAYSCALE

	void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}



	void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform(src, dst, conv);
	}


	void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_self(src_dst, conv);
	}


	void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);
		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
		transform_self(src_dst, conv);
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL



	namespace seq
	{
#ifndef LIBIMAGE_NO_GRAYSCALE

		void contrast(gray::image_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::image_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::image_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform(src, dst, conv);
		}


		void contrast_self(gray::image_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform_self(src_dst, conv);
		}


		void contrast_self(gray::view_t const& src_dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);
			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, U8_MIN, U8_MAX, p); };
			seq::transform_self(src_dst, conv);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}

}


/*  convolve.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
	static inline void left_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left (2 wide)
		assert(x <= width - 2);
		range.x_begin = x;
		range.x_end = x + 2;
	}


	static inline void right_2_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// right (2 wide)
		assert(x >= 1);
		assert(x <= width - 1);
		range.x_begin = x - 1;
		range.x_end = x + 1;
	}


	static inline void top_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top (2 high)
		assert(y <= height - 2);
		range.y_begin = y;
		range.y_end = y + 2;
	}


	static inline void bottom_2_high(pixel_range_t& range, u32 y, u32 height)
	{
		// bottom (2 high)
		assert(y >= 1);
		assert(y <= height - 1);
		range.y_begin = y - 1;
		range.y_end = y + 1;
	}


	static inline void top_or_bottom_3_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (3 high)
		assert(y >= 1);
		assert(y <= height - 2);
		range.y_begin = y - 1;
		range.y_end = y + 2;
	}


	static inline void left_or_right_3_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (3 wide)
		assert(x >= 1);
		assert(x <= width - 2);
		range.x_begin = x - 1;
		range.x_end = x + 2;
	}


	static inline void top_or_bottom_5_high(pixel_range_t& range, u32 y, u32 height)
	{
		// top or bottom (5 high)
		assert(y >= 2);
		assert(y <= height - 3);
		range.y_begin = y - 2;
		range.y_end = y + 3;
	}


	static inline void left_or_right_5_wide(pixel_range_t& range, u32 x, u32 width)
	{
		// left or right (5 wide)
		assert(x >= 2);
		assert(x <= width - 3);
		range.x_begin = x - 2;
		range.x_end = x + 3;
	}


	template<size_t N>
	static r32 apply_weights(gray::view_t const& view, std::array<r32, N> const& weights)
	{
		assert((size_t)(view.width) * view.height == weights.size());

		u32 w = 0;
		r32 total = 0.0f;

		auto const add_weight = [&](u8 p)
		{
			total += weights[w++] * p;
		};

		for_each_pixel(view, add_weight);

		return total;
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 9> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_center(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 25> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_5_high(range, y, img.height);

		left_or_right_5_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 4> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_top(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_2_high(range, y, img.height);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_bottom(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		bottom_2_high(range, y, img.width);

		left_or_right_3_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_left(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		left_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
	}


	template<class GRAY_IMAGE_T>
	static r32 weighted_right(GRAY_IMAGE_T const& img, u32 x, u32 y, std::array<r32, 6> const& weights)
	{
		pixel_range_t range = {};

		top_or_bottom_3_high(range, y, img.height);

		right_2_wide(range, x, img.width);

		return apply_weights(sub_view(img, range), weights);
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


	u8 gauss3(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss3(gray::view_t const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_3X3);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::image_t const& img, u32 x, u32 y)
	{
		auto p = weighted_center(img, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	u8 gauss5(gray::view_t const& view, u32 x, u32 y)
	{
		auto p = weighted_center(view, x, y, GAUSS_5X5);

		assert(p >= 0.0f);
		assert(p <= 255.0f);

		return (u8)(p);
	}


	r32 x_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_X_3X3);
	}


	r32 x_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_X_3X3);
	}


	r32 y_gradient(gray::image_t const& img, u32 x, u32 y)
	{
		return weighted_center(img, x, y, GRAD_Y_3X3);
	}


	r32 y_gradient(gray::view_t const& view, u32 x, u32 y)
	{
		return weighted_center(view, x, y, GRAD_Y_3X3);
	}


#ifndef LIBIMAGE_NO_SIMD



#endif // !LIBIMAGE_NO_SIMD
}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  blur.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


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
		pixel_range_t r = {};
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
		pixel_range_t r = {};
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


	void blur(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


	void blur(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));
		auto const width = src.width;
		auto const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		do_blur(src, dst);
	}


#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void copy_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = src.width - 1;
			u32 y_last = src.height - 1;
			auto src_top = src.row_begin(y_first);
			auto src_bottom = src.row_begin(y_last);
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = src_top[x];
				dst_bottom[x] = src_bottom[x];
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void copy_left_right(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 1;
			u32 x_last = src.width - 1;
			u32 y_last = src.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto src_row = src.row_begin(y);
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = src_row[x_first];
				dst_row[x_last] = src_row[x_last];
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = src.width - 2;
			u32 y_last = src.height - 2;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = gauss3(src, x, y_first);
				dst_bottom[x] = gauss3(src, x, y_last);
			}
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_left_right(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 2;
			u32 x_last = src.width - 2;
			u32 y_last = src.height - 3;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = gauss3(src, x_first, y);
				dst_row[x_last] = gauss3(src, x_last, y);
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void inner_gauss(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_first = 2;
			u32 y_first = 2;
			u32 x_last = src.width - 3;
			u32 y_last = src.height - 3;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);

				for (u32 x = x_first; x <= x_last; ++x)
				{
					dst_row[x] = gauss5(src, x, y);
				}
			}
		}


		void blur(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			seq::inner_gauss(src, dst);
		}
	}

#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{
		void blur(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}


		void blur(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			assert(src.width >= VIEW_MIN_DIM);
			assert(src.height >= VIEW_MIN_DIM);

			seq::copy_top_bottom(src, dst);
			seq::copy_left_right(src, dst);
			seq::gauss_inner_top_bottom(src, dst);
			seq::gauss_inner_left_right(src, dst);
			simd::inner_gauss(src, dst);
		}
	}

#endif // !LIBIMAGE_NO_SIMD

}

#endif // !LIBIMAGE_NO_GRAYSCALE


/*  edges_gradients.cpp  */

#ifndef LIBIMAGE_NO_GRAYSCALE

namespace libimage
{
#ifndef LIBIMAGE_NO_PARALLEL


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
		pixel_range_t r = {};
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
		pixel_range_t r = {};
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


	void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_edges(temp, dst, cond);
	}


	void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
	{
		assert(verify(src, dst));

		do_edges(src, dst, cond);
	}


	void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
	{
		assert(verify(src, dst));
		assert(verify(src, temp));

		blur(src, temp);

		do_gradients(temp, dst);
	}


	void gradients(gray::image_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::image_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::view_t const& src, gray::image_t const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


	void gradients(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		do_gradients(src, dst);
	}


#endif // !LIBIMAGE_NO_PARALLEL

	namespace seq
	{
		template<class GRAY_IMG_T>
		static void zero_top_bottom(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 0;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 1;
			auto dst_top = dst.row_begin(y_first);
			auto dst_bottom = dst.row_begin(y_last);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_top[x] = 0;
				dst_bottom[x] = 0;
			}
		}


		template<class GRAY_IMG_T>
		static void zero_left_right(GRAY_IMG_T const& dst)
		{
			u32 x_first = 0;
			u32 y_first = 1;
			u32 x_last = dst.width - 1;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				dst_row[x_first] = 0;
				dst_row[x_last] = 0;
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = dst.width - 2;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				for (u32 x = x_first; x <= x_last; ++x)
				{
					auto gx = x_gradient(src, x, y);
					auto gy = y_gradient(src, x, y);
					auto g = (u8)(std::hypot(gx, gy));
					dst_row[x] = cond(g) ? 255 : 0;
				}
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			zero_top_bottom(dst);
			zero_left_right(dst);

			seq::edges_inner(src, dst, cond);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_first = 1;
			u32 y_first = 1;
			u32 x_last = dst.width - 2;
			u32 y_last = dst.height - 2;
			for (u32 y = y_first; y <= y_last; ++y)
			{
				auto dst_row = dst.row_begin(y);
				for (u32 x = x_first; x <= x_last; ++x)
				{
					auto gx = x_gradient(src, x, y);
					auto gy = y_gradient(src, x, y);
					auto g = std::hypot(gx, gy);
					dst_row[x] = (u8)(g);
				}
			}
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			seq::gradients_inner(src, dst);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			seq::do_edges(src, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			seq::do_edges(src, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			seq::do_edges(src, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			seq::do_edges(src, dst, cond);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_edges(temp, dst, cond);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			seq::do_gradients(src, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			seq::do_gradients(src, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			seq::do_gradients(src, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			seq::do_gradients(src, dst);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			seq::blur(src, temp);

			seq::do_gradients(temp, dst);
		}
	}


#ifndef LIBIMAGE_NO_SIMD

	namespace simd
	{

		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_edges(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			simd::inner_edges(src, dst, cond);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void do_gradients(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			seq::zero_top_bottom(dst);
			seq::zero_left_right(dst);

			simd::inner_gradients(src, dst);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_edges(temp, dst, cond);
		}


		void edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			simd::do_edges(src, dst, cond);
		}


		void edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			simd::do_edges(src, dst, cond);
		}


		void edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			simd::do_edges(src, dst, cond);
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			assert(verify(src, dst));

			simd::do_edges(src, dst, cond);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}



		void gradients(gray::view_t const& src, gray::view_t const& dst, gray::image_t const& temp)
		{
			assert(verify(src, dst));
			assert(verify(src, temp));

			simd::blur(src, temp);
			simd::do_gradients(temp, dst);
		}


		void gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			simd::do_gradients(src, dst);
		}


		void gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			simd::do_gradients(src, dst);
		}


		void gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			assert(verify(src, dst));

			simd::do_gradients(src, dst);
		}


		void gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			simd::do_gradients(src, dst);
		}

	}

#endif // !LIBIMAGE_NO_SIMD

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
	static pixel_t get_color(IMG_T const& src_image, Point2Dr32 location)
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


#ifndef LIBIMAGE_NO_PARALLEL

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


	void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin_x, origin_y, theta);
	}


	void rotate(image_t const& src, image_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(image_t const& src, view_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(view_t const& src, image_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par(src, dst, origin.x, origin.y, theta);
	}


	void rotate(view_t const& src, view_t const& dst, Point2Du32 origin, r32 theta)
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


	void rotate(gray::image_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::image_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::view_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::view_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin_x, origin_y, theta);
	}


	void rotate(gray::image_t const& src, gray::image_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::image_t const& src, gray::view_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::view_t const& src, gray::image_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


	void rotate(gray::view_t const& src, gray::view_t const& dst, Point2Du32 origin, r32 theta)
	{
		assert(verify(src));
		assert(verify(dst));

		rotate_par_gray(src, dst, origin.x, origin.y, theta);
	}


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR


		template <typename IMG_SRC_T, typename IMG_DST_T>
		static void rotate_seq(IMG_SRC_T const& src, IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			Point2Du32 origin = { origin_x, origin_y };

			auto const func = [&](u32 x, u32 y)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, theta);
				*dst.xy_at(x, y) = get_color(src, src_pt);
			};

			for_each_xy(dst, func);
		}


		void rotate(image_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(image_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(view_t const& src, image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(view_t const& src, view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin_x, origin_y, theta);
		}


		void rotate(image_t const& src, image_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin.x, origin.y, theta);
		}


		void rotate(image_t const& src, view_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin.x, origin.y, theta);
		}


		void rotate(view_t const& src, image_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin.x, origin.y, theta);
		}


		void rotate(view_t const& src, view_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq(src, dst, origin.x, origin.y, theta);
		}


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		template <typename GR_IMG_SRC_T, typename GR_IMG_DST_T>
		static void rotate_seq_gray(GR_IMG_SRC_T const& src, GR_IMG_DST_T const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			Point2Du32 origin = { origin_x, origin_y };

			auto const func = [&](u32 x, u32 y)
			{
				auto src_pt = find_rotation_src({ x, y }, origin, theta);
				*dst.xy_at(x, y) = get_gray(src, src_pt);
			};

			for_each_xy(dst, func);
		}


		void rotate(gray::image_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::image_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::view_t const& src, gray::image_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::view_t const& src, gray::view_t const& dst, u32 origin_x, u32 origin_y, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin_x, origin_y, theta);
		}


		void rotate(gray::image_t const& src, gray::image_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin.x, origin.y, theta);
		}


		void rotate(gray::image_t const& src, gray::view_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin.x, origin.y, theta);
		}


		void rotate(gray::view_t const& src, gray::image_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin.x, origin.y, theta);
		}


		void rotate(gray::view_t const& src, gray::view_t const& dst, Point2Du32 origin, r32 theta)
		{
			assert(verify(src));
			assert(verify(dst));

			rotate_seq_gray(src, dst, origin.x, origin.y, theta);
		}


#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}