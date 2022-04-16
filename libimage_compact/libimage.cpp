#include "libimage.hpp"

#ifndef LIBIMAGE_NO_PARALLEL
#include <execution>
#endif // !LIBIMAGE_NO_PARALLEL

#include <algorithm>


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