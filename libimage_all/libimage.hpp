#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

//#define LIBIMAGE_NO_COLOR
//#define LIBIMAGE_NO_GRAYSCALE
//#define LIBIMAGE_NO_WRITE
//#define LIBIMAGE_NO_RESIZE
//#define LIBIMAGE_NO_FS

#include <cstdint>
#include <iterator>
#include <cassert>

#ifndef LIBIMAGE_NO_FS
#include <filesystem>
namespace fs = std::filesystem;
#endif // !LIBIMAGE_NO_FS

#ifndef LIBIMAGE_NO_COLOR
#include <functional>
#endif // !LIBIMAGE_NO_COLOR

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;

namespace libimage
{
	//======= Class Definitions =============

	// region of interest in an image
	typedef struct
	{
		u32 x_begin;
		u32 x_end;   // one past last x
		u32 y_begin;
		u32 y_end;   // one past last y

	} pixel_range_t;


#ifndef LIBIMAGE_NO_COLOR

	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;
	
	enum class Channel : u32
	{
		Red = 0,
		Green,
		Blue,
		Alpha
	};


	inline u32 to_channel_index(Channel ch)
	{
		return static_cast<u32>(ch);
	}


	inline void for_each_channel_rgb(std::function<void(u32 ch_id)> const& func)
	{
		for (u32 id = 0; id < RGB_CHANNELS; ++id)
		{
			func(id);
		}
	}


	inline void for_each_channel_rgba(std::function<void(u32 ch_id)> const& func)
	{
		for (u32 id = 0; id < RGBA_CHANNELS; ++id)
		{
			func(id);
		}
	}


	// color pixel
	typedef union rgba_pixel_t
	{
		struct
		{
			u8 red;
			u8 green;
			u8 blue;
			u8 alpha;
		};

		u8 channels[RGBA_CHANNELS];

		u32 value;

	}rgba_pixel;

	using pixel_t = rgba_pixel;


	// color image
	// owns the memory
	class rgba_image_t
	{
	public:
		u32 width = 0;
		u32 height = 0;

		pixel_t* data = 0;

		void clear()
		{
			if (data)
			{
				free(data);
			}
		}

		~rgba_image_t()
		{
			clear();
		}

		rgba_pixel_t* begin() { return data; }
		rgba_pixel_t* end() { return data + (u64)width * (u64)height; }
		rgba_pixel_t* begin() const { return data; }
		rgba_pixel_t* end() const { return data + (u64)width * (u64)height; }
	};

	using image_t = rgba_image_t;


	// subset of existing image data
	class rgba_image_view_t
	{
	public:

		pixel_t* image_data = 0;
		u32 image_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;

		pixel_t* row_begin(u32 y) const
		{
			assert(y < height);

			auto offset = (y_begin + y) * image_width + x_begin;

			auto ptr = image_data + static_cast<u64>(offset);
			assert(ptr);

			return ptr;
		}

		pixel_t* xy_at(u32 x, u32 y) const
		{
			assert(y < height);
			assert(x < width);
			return row_begin(y) + x;
		}


		/******* ITERATOR ************/

		class iterator
		{
		private:

			u32 loc_x = 0;
			u32 loc_y = 0;

			u32 x_begin = 0;
			u32 x_end = 0;
			u32 y_begin = 0;
			u32 y_end = 0;

			pixel_t* image_data = 0;
			u32 image_width = 0;

			pixel_t* loc_ptr() const
			{
				assert(loc_x >= x_begin);
				assert(loc_x < x_end);
				assert(loc_y >= y_begin);
				assert(loc_y < y_end);

				auto offset = loc_y * image_width + loc_x;
				auto ptr = image_data + static_cast<u64>(offset);
				assert(ptr);

				return ptr;
			}

			void next()
			{
				++loc_x;
				if (loc_x >= x_end)
				{
					loc_x = x_begin;
					++loc_y;
				}

				assert(loc_x >= x_begin);
				assert(loc_x <= x_end);
				assert(loc_y >= y_begin);
				assert(loc_y <= y_end);
			}

		public:

			using iterator_category = std::forward_iterator_tag;
			using value_type = pixel_t;
			using difference_type = std::ptrdiff_t;
			using pointer = value_type*;
			using reference = value_type&;

			explicit iterator() {}

			explicit iterator(rgba_image_view_t const& view)
			{
				image_data = view.image_data;
				image_width = view.image_width;

				x_begin = view.x_begin;
				x_end = view.x_end;
				y_begin = view.y_begin;
				y_end = view.y_end;

				loc_x = x_begin;
				loc_y = y_begin;
			}

			iterator end()
			{
				loc_x = x_end - 1;
				loc_y = y_end - 1;

				next();

				return *this;
			}

			iterator& operator ++ ()
			{
				next();

				return *this;
			}

			iterator operator ++ (int) { iterator result = *this; ++(*this); return result; }

			bool operator == (iterator other) const { return loc_x == other.loc_x && loc_y == other.loc_y; }

			bool operator != (iterator other) const { return !(*this == other); }

			reference operator * () const { return *loc_ptr(); }
		};
		/* ^^^^^^^^^ ITERATOR ^^^^^^^^^^ */


		iterator begin() { return iterator(*this); }

		iterator end() { return iterator(*this).end(); }

		iterator begin() const { return iterator(*this); }

		iterator end() const { return iterator(*this).end(); }

	};

	using view_t = rgba_image_view_t;


	constexpr pixel_t to_pixel(u8 red, u8 green, u8 blue, u8 alpha)
	{
		pixel_t pixel{};
		pixel.red = red;
		pixel.green = green;
		pixel.blue = blue;
		pixel.alpha = alpha;

		return pixel;
	}


	constexpr pixel_t to_pixel(u8 red, u8 green, u8 blue)
	{
		return to_pixel(red, green, blue, 255);
	}


	constexpr pixel_t to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	namespace gray
	{
		// grayscale value as an unsigned 8-bit integer
		using pixel_t = u8;


		// grayscale image
		class image_t
		{
		public:
			u32 width = 0;
			u32 height = 0;

			pixel_t* data = 0;

			void clear()
			{
				if (data)
				{
					free(data);
				}
			}

			~image_t()
			{
				clear();
			}

			pixel_t* begin() { return data; }

			pixel_t* end() { return data + (u64)width * (u64)height; }

			pixel_t* begin() const { return data; }

			pixel_t* end() const { return data + (u64)width * (u64)height; }
		};


		// subset of grayscale image data
		class image_view_t
		{
		public:

			pixel_t* image_data = 0;
			u32 image_width = 0;

			u32 x_begin = 0;
			u32 x_end = 0;
			u32 y_begin = 0;
			u32 y_end = 0;

			u32 width = 0;
			u32 height = 0;

			pixel_t* row_begin(u32 y) const
			{
				auto offset = (y_begin + y) * image_width + x_begin;
				return image_data + (u64)offset;
			}

			pixel_t* xy_at(u32 x, u32 y) const
			{
				return row_begin(y) + x;
			}

			/******* ITERATOR ************/

			class iterator
			{
			private:

				u32 loc_x = 0;
				u32 loc_y = 0;

				u32 x_begin = 0;
				u32 x_end = 0;
				u32 y_begin = 0;
				u32 y_end = 0;

				pixel_t* image_data = 0;
				u32 image_width = 0;

				pixel_t* loc_ptr() const
				{
					assert(loc_x >= x_begin);
					assert(loc_x < x_end);
					assert(loc_y >= y_begin);
					assert(loc_y < y_end);

					auto offset = loc_y * image_width + loc_x;
					auto ptr = image_data + static_cast<u64>(offset);
					assert(ptr);

					return ptr;
				}

				void next()
				{
					++loc_x;
					if (loc_x >= x_end)
					{
						loc_x = x_begin;
						++loc_y;
					}

					assert(loc_x >= x_begin);
					assert(loc_x <= x_end);
					assert(loc_y >= y_begin);
					assert(loc_y <= y_end);
				}

			public:

				using iterator_category = std::forward_iterator_tag;
				using value_type = pixel_t;
				using difference_type = std::ptrdiff_t;
				using pointer = value_type*;
				using reference = value_type&;

				explicit iterator() {}

				explicit iterator(image_view_t const& view)
				{
					image_data = view.image_data;
					image_width = view.image_width;

					x_begin = view.x_begin;
					x_end = view.x_end;
					y_begin = view.y_begin;
					y_end = view.y_end;

					loc_x = x_begin;
					loc_y = y_begin;
				}

				iterator end()
				{
					loc_x = x_end - 1;
					loc_y = y_end - 1;
					next();

					return *this;
				}

				iterator& operator ++ ()
				{
					next();

					return *this;
				}

				iterator operator ++ (int) { iterator result = *this; ++(*this); return result; }

				bool operator == (iterator other) const { return loc_x == other.loc_x && loc_y == other.loc_y; }

				bool operator != (iterator other) const { return !(*this == other); }

				reference operator * () const { return *loc_ptr(); }
			};
			/* ^^^^^^^^^ ITERATOR ^^^^^^^^^^ */

			iterator begin() { return iterator(*this); }

			iterator end() { return iterator(*this).end(); }

			iterator begin() const { return iterator(*this); }

			iterator end() const { return iterator(*this).end(); }
		};

		using view_t = image_view_t;


	}

	namespace grey = gray;

#endif // !LIBIMAGE_NO_GRAYSCALE


	//======= Functions ==================

#ifndef LIBIMAGE_NO_COLOR

	void read_image_from_file(const char* img_path_src, image_t& image_dst);

	void make_image(image_t& image_dst, u32 width, u32 height);

	view_t make_view(image_t const& image);

	view_t sub_view(image_t const& image, pixel_range_t const& range);

	view_t sub_view(view_t const& view, pixel_range_t const& range);

	view_t row_view(image_t const& image, u32 y);

	view_t row_view(view_t const& view, u32 y);

	view_t column_view(image_t const& image, u32 x);

	view_t column_view(view_t const& view, u32 x);

	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y);

	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y);

	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x);

	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x);

#ifndef LIBIMAGE_NO_WRITE

	void write_image(image_t const& image_src, const char* file_path_dst);

	void write_view(view_t const& view_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(image_t const& image_src, image_t& image_dst);

	view_t make_resized_view(image_t const& image_src, image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

	void read_image_from_file(const char* file_path_src, gray::image_t& image_dst);

	void make_image(gray::image_t& image_dst, u32 width, u32 height);

	gray::view_t make_view(gray::image_t const& image);

	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range);

	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range);

	gray::view_t row_view(gray::image_t const& image, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y);

	gray::view_t column_view(gray::view_t const& view, u32 y);

	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x);

	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x);


#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::image_t const& image_src, const char* file_path_dst);

	void write_view(gray::view_t const& view_src, const char* file_path_dst);

#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::image_t const& img_src, gray::image_t& img_dst);

	gray::view_t make_resized_view(gray::image_t const& image_src, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_FS

#ifndef LIBIMAGE_NO_COLOR

	inline void read_image_from_file(fs::path const& img_path_src, image_t& image_dst)
	{
		auto file_path_str = img_path_src.string();

		read_image_from_file(file_path_str.c_str(), image_dst);
	}


	inline void write_image(image_t const& image_src, fs::path const& file_path)
	{
		auto file_path_str = file_path.string();

		write_image(image_src, file_path_str.c_str());
	}


	inline void write_view(view_t const& view_src, fs::path const& file_path)
	{
		auto file_path_str = file_path.string();

		write_view(view_src, file_path_str.c_str());
	}
#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	inline void read_image_from_file(fs::path const& img_path_src, gray::image_t& image_dst)
	{
		auto file_path_str = img_path_src.string();

		return read_image_from_file(file_path_str.c_str(), image_dst);
	}


	inline void write_image(gray::image_t const& image_src, fs::path const& file_path_dst)
	{
		auto file_path_str = file_path_dst.string();

		write_image(image_src, file_path_str.c_str());
	}


	inline void write_view(gray::view_t const& view_src, fs::path const& file_path_dst)
	{
		auto file_path_str = file_path_dst.string();

		write_view(view_src, file_path_str.c_str());
	}
#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_FS

}
