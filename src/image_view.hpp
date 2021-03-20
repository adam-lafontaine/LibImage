#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include <cstdint>
#include <functional>
#include <iterator>
#include <cassert>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

namespace libimage
{
	constexpr auto RGBA_CHANNELS = 4u;

	// region of interest in an image
	typedef struct
	{
		u32 x_begin;
		u32 x_end;   // one past last x
		u32 y_begin;
		u32 y_end;   // one past last y

	} pixel_range_t;


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

		~rgba_image_t()
		{
			if (data)
			{
				free(data);
			}
		}

		rgba_pixel_t* begin() { return data; }
		rgba_pixel_t* end() { return data + (u64)width * (u64)height; }
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

			/*void next_in_row()
			{
				++loc_x;
			}

			void next_in_column()
			{
				++loc_y;
			}

			std::function<void()> increment = [&]() { next(); };*/

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

				/*if (y_end - y_begin == 1)
				{
					increment = [&]() { next_in_row(); };
				}
				else if (x_end - x_begin == 1)
				{
					increment = [&]() { next_in_column(); };
				}*/
			}

			iterator end()
			{
				loc_x = x_end - 1;
				loc_y = y_end - 1;

				//increment();
				next();

				return *this;
			}

			iterator& operator ++ ()
			{
				//increment();

				next();

				return *this;
			}

			iterator operator ++ (int) { iterator result = *this; ++(*this); return result; }

			bool operator == (iterator other) const { return loc_x == other.loc_x && loc_y == other.loc_y; }

			bool operator != (iterator other) const { return !(*this == other); }

			reference operator * () const { return *loc_ptr(); }
		};

		
		/******* ITERATOR ************/

		iterator begin() { return iterator(*this); }

		iterator end() { return iterator(*this).end(); }

		iterator cbegin() const { return iterator(*this); }

		iterator cend() const { return iterator(*this).end(); }

	};

	using view_t = rgba_image_view_t;


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

			~image_t()
			{
				if (data)
				{
					free(data);
				}
			}

			pixel_t* begin() { return data; }
			pixel_t* end() { return data + (u64)width * (u64)height; }
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

				/*void next_in_row()
				{
					++loc_x;
				}

				void next_in_column()
				{
					++loc_y;
				}

				std::function<void()> increment = [&]() { next(); };*/

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

					/*if (y_begin == y_end)
					{
						increment = [&]() { next_in_row(); };
					}
					else if (x_begin == x_end)
					{
						increment = [&]() { next_in_column(); };
					}*/
				}

				iterator end()
				{
					loc_x = x_end - 1;
					loc_y = y_end - 1;

					//increment();
					next();

					return *this;
				}

				iterator& operator ++ ()
				{
					//increment();

					next();

					return *this;
				}

				iterator operator ++ (int) { iterator result = *this; ++(*this); return result; }

				bool operator == (iterator other) const { return loc_x == other.loc_x && loc_y == other.loc_y; }

				bool operator != (iterator other) const { return !(*this == other); }

				reference operator * () const { return *loc_ptr(); }
			};

			/******* ITERATOR ************/

			iterator begin() { return iterator(*this); }

			iterator end() { return iterator(*this).end(); }

			iterator cbegin() const { return iterator(*this); }

			iterator cend() const { return iterator(*this).end(); }
		};

		using view_t = image_view_t;
	}
	
	namespace grey = gray;

	inline pixel_t to_pixel(u8 red, u8 green, u8 blue, u8 alpha)
	{
		pixel_t pixel{};
		pixel.red = red;
		pixel.green = green;
		pixel.blue = blue;
		pixel.alpha = alpha;

		return pixel;

		//return { red, green, blue, alpha };
	}


	inline pixel_t to_pixel(u8 red, u8 green, u8 blue)
	{
		return to_pixel(red, green, blue, 255);
	}


	inline pixel_t to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}

}
