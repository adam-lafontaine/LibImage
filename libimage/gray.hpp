#pragma once

#include "types.hpp"

#include <iterator>
#include <cassert>


namespace libimage
{
	namespace gray
	{
		// grayscale value as an unsigned 8-bit integer
		using pixel_t = u8;


		// grayscale image
		class Image
		{
		public:
			u32 width = 0;
			u32 height = 0;

			pixel_t* data = nullptr;

			pixel_t* row_begin(u32 y) const
			{
				assert(y < height);

				auto offset = y * width;

				auto ptr = data + static_cast<u64>(offset);
				assert(ptr);

				return ptr;
			}

			pixel_t* xy_at(u32 x, u32 y) const
			{
				assert(y < height);
				assert(x < width);
				return row_begin(y) + x;
			}

			void dispose()
			{
				if (data != nullptr)
				{
					free(data);
					data = nullptr;
				}
			}

			~Image()
			{
				dispose();
			}



			pixel_t* begin() { return data; }

			pixel_t* end() { return data + static_cast<u64>(width * static_cast<u64>(height)); }

			pixel_t* begin() const { return data; }

			pixel_t* end() const { return data + static_cast<u64>(width * static_cast<u64>(height)); }
		};


		// subset of grayscale image data
		class ImageView
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
				return image_data + static_cast<u64>(offset);
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

				explicit iterator(ImageView const& view)
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

				xy_loc_t get_xy() { return { loc_x, loc_y }; }

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

		


		class ImageRowView
		{
		public:

			pixel_t* image_data = 0;
			u32 image_width = 0;

			u32 x_begin = 0;
			u32 x_end = 0;
			u32 y_begin = 0;

			u32 width = 0;

			pixel_t* begin() { return image_data + static_cast<u64>(y_begin * image_width + x_begin); }

			pixel_t* end() { return image_data + static_cast<u64>(y_begin * image_width + x_end); }

			pixel_t* begin() const { return image_data + static_cast<u64>(y_begin * image_width + x_begin); }

			pixel_t* end() const { return image_data + static_cast<u64>(y_begin * image_width + x_end); }
		};


		class ImageColumnView
		{
		public:

			pixel_t* image_data = 0;
			u32 image_width = 0;

			u32 x_begin = 0;			
			u32 y_begin = 0;
			u32 y_end = 0;

			class iterator
			{
			private:

				u32 loc_x = 0;
				u32 loc_y = 0;

				u32 y_begin = 0;
				u32 y_end = 0;

				pixel_t* image_data = 0;
				u32 image_width = 0;

				pixel_t* loc_ptr() const
				{
					assert(loc_y >= y_begin);
					assert(loc_y < y_end);

					auto offset = loc_y * image_width + loc_x;
					auto ptr = image_data + static_cast<u64>(offset);
					assert(ptr);

					return ptr;
				}

				void next()
				{
					++loc_y;

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

				explicit iterator(ImageColumnView const& view)
				{
					image_data = view.image_data;
					image_width = view.image_width;

					y_begin = view.y_begin;
					y_end = view.y_end;

					loc_x = view.x_begin;
					loc_y = y_begin;
				}

				xy_loc_t get_xy() { return { loc_x, loc_y }; }

				iterator end()
				{
					loc_y = y_end;

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

			iterator begin() { return iterator(*this); }

			iterator end() { return iterator(*this).end(); }

			iterator begin() const { return iterator(*this); }

			iterator end() const { return iterator(*this).end(); }
		};


		using image_t = Image;
		using view_t = ImageView;
		using row_view_t = ImageRowView;
		using column_view_t = ImageColumnView;
	}
}