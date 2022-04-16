#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <iterator>
#include <cassert>
#include <functional>

#include "defines.hpp"


// TODO: LIBIMAGE_NO_RAII !!!


/*  types.hpp  */

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;
using i32 = int32_t;


class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Point2Dr32
{
public:
	r32 x;
	r32 y;
};

namespace libimage
{
	// region of interest in an image
	typedef struct
	{
		u32 x_begin;
		u32 x_end;   // one past last x
		u32 y_begin;
		u32 y_end;   // one past last y

	} pixel_range_t;


	typedef struct
	{
		u32 x;
		u32 y;

	} xy_loc_t;


	constexpr size_t CHANNEL_SIZE = 256; // 8 bit channel
	constexpr size_t N_HIST_BUCKETS = 256;

	using hist_t = std::array<u32, N_HIST_BUCKETS>;


	typedef struct channel_stats_t
	{
		r32 mean;
		r32 std_dev;
		hist_t hist;

	} stats_t;


	typedef union rgb_channel_stats_t
	{
		struct
		{
			stats_t red;
			stats_t green;
			stats_t blue;
		};

		stats_t stats[3];

	} rgb_stats_t;
}

#ifndef LIBIMAGE_NO_COLOR

/* rgba.hpp  */

namespace libimage
{
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
		return (u32)(ch);
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


	// color image
	// owns the memory
	class RGBAImage
	{
	public:
		u32 width = 0;
		u32 height = 0;

		pixel_t* data = nullptr;

		pixel_t* row_begin(u32 y) const
		{
			assert(y < height);

			auto offset = y * width;

			auto ptr = data + (u64)(offset);
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

		~RGBAImage()
		{
			dispose();
		}

		rgba_pixel_t* begin() { return data; }
		rgba_pixel_t* end() { return data + (u64)(width) * (u64)(height); }
		rgba_pixel_t* begin() const { return data; }
		rgba_pixel_t* end() const { return data + (u64)(width) * (u64)(height); }
	};


	// subset of existing image data
	class RGBAImageView
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

			auto ptr = image_data + (u64)(offset);
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
				auto ptr = image_data + (u64)(offset);
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

			explicit iterator(RGBAImageView const& view)
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


			xy_loc_t get_xy() { return { loc_x, loc_y }; }


			iterator& operator ++ () { next(); return *this; }

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


	class RGBAImageRowView
	{
	public:

		pixel_t* image_data = 0;
		u32 image_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;

		u32 width = 0;

		pixel_t* begin() { return image_data + (u64)(y_begin)*image_width + x_begin; }

		pixel_t* end() { return image_data + (u64)(y_begin)*image_width + x_end; }

		pixel_t* begin() const { return image_data + (u64)(y_begin)*image_width + x_begin; }

		pixel_t* end() const { return image_data + (u64)(y_begin)*image_width + x_end; }
	};


	class RGBAImageColumnView
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
				auto ptr = image_data + (u64)(offset);
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

			explicit iterator(RGBAImageColumnView const& view)
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

			iterator& operator ++ () { next(); return *this; }

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


	using image_t = RGBAImage;
	using view_t = RGBAImageView;
	using row_view_t = RGBAImageRowView;
	using column_view_t = RGBAImageColumnView;


	class RGBAPlanar
	{
	public:
		u32 width;
		u32 height;

		u8* data;

		u8* red;
		u8* green;
		u8* blue;
		u8* alpha;

		void dispose()
		{
			if (data)
			{
				free(data);
				data = nullptr;
			}
		}

		~RGBAPlanar()
		{
			dispose();
		}
	};


	using image_soa = RGBAPlanar;
}

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE

/*  gray.hpp  */

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
				assert(width);
				assert(height);
				assert(data);
				assert(y < height);

				auto offset = y * width;

				auto ptr = data + (u64)(offset);
				assert(ptr);

				return ptr;
			}

			pixel_t* xy_at(u32 x, u32 y) const
			{
				assert(width);
				assert(height);
				assert(data);
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

			pixel_t* end() { return data + (u64)(width * (u64)(height)); }

			pixel_t* begin() const { return data; }

			pixel_t* end() const { return data + (u64)(width * (u64)(height)); }
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
				return image_data + (u64)(offset);
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
					auto ptr = image_data + (u64)(offset);
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

				iterator& operator ++ () { next(); return *this; }

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

			pixel_t* begin() { return image_data + (u64)(y_begin)*image_width + x_begin; }

			pixel_t* end() { return image_data + (u64)(y_begin)*image_width + x_end; }

			pixel_t* begin() const { return image_data + (u64)(y_begin)*image_width + x_begin; }

			pixel_t* end() const { return image_data + (u64)(y_begin)*image_width + x_end; }
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
					auto ptr = image_data + (u64)(offset);
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

				iterator& operator ++ () { next(); return *this; }

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


#endif // !LIBIMAGE_NO_GRAYSCALE


/*  libimage.hpp  */

namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void read_image_from_file(const char* img_path_src, image_t& image_dst);

	void make_image(image_t& image_dst, u32 width, u32 height);

	view_t make_view(image_t const& image);

	view_t make_view(image_t& image, u32 width, u32 height);

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

	void for_each_pixel(image_t const& image, std::function<void(pixel_t& p)> const& func);

	void for_each_pixel(view_t const& view, std::function<void(pixel_t& p)> const& func);

	void for_each_xy(image_t const& image, std::function<void(u32 x, u32 y)> const& func);

	void for_each_xy(view_t const& view, std::function<void(u32 x, u32 y)> const& func);


	void make_planar(image_soa& dst, u32 width, u32 height);


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

	gray::view_t make_view(gray::image_t& image, u32 width, u32 height);

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

	void for_each_pixel(gray::image_t const& image, std::function<void(gray::pixel_t& p)> const& func);

	void for_each_pixel(gray::view_t const& view, std::function<void(gray::pixel_t& p)> const& func);

	void for_each_xy(gray::image_t const& image, std::function<void(u32 x, u32 y)> const& func);

	void for_each_xy(gray::view_t const& view, std::function<void(u32 x, u32 y)> const& func);


#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::image_t const& image_src, const char* file_path_dst);

	void write_view(gray::view_t const& view_src, const char* file_path_dst);


#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::image_t const& img_src, gray::image_t& img_dst);

	gray::view_t make_resized_view(gray::image_t const& image_src, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_GRAYSCALE

}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	inline void read_image_from_file(std::string const& img_path_src, image_t& image_dst)
	{
		read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(image_t const& image_src, std::string const& file_path)
	{
		write_image(image_src, file_path.c_str());
	}

	inline void write_view(view_t const& view_src, std::string const& file_path)
	{
		write_view(view_src, file_path.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	inline void read_image_from_file(std::string const& img_path_src, gray::image_t& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::image_t const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}

	inline void write_view(gray::view_t const& view_src, std::string const& file_path_dst)
	{
		write_view(view_src, file_path_dst.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}


#ifndef LIBIMAGE_NO_FILESYSTEM

#include <filesystem>
namespace fs = std::filesystem;


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

	inline void read_image_from_file(fs::path const& img_path_src, image_t& image_dst)
	{
		read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(image_t const& image_src, fs::path const& file_path)
	{
		write_image(image_src, file_path.string().c_str());
	}

	inline void write_view(view_t const& view_src, fs::path const& file_path)
	{
		write_view(view_src, file_path.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	inline void read_image_from_file(fs::path const& img_path_src, gray::image_t& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::image_t const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

	inline void write_view(gray::view_t const& view_src, fs::path const& file_path_dst)
	{
		write_view(view_src, file_path_dst.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}

#endif // !LIBIMAGE_NO_FILESYSTEM


/*  process.hpp  */

namespace libimage
{
	constexpr u32 VIEW_MIN_DIM = 5;

}


/*  proc_def.hpp  */

namespace libimage
{
	using u8_to_bool_f = std::function<bool(u8)>;

	using pixel_to_bool_f = std::function<bool(pixel_t)>;

	using pixel_to_pixel_f = std::function<pixel_t(pixel_t const&)>;

	using pixel_to_u8_f = std::function<u8(pixel_t const& p)>;

	using u8_to_u8_f = std::function<u8(u8)>;

	using lookup_table_t = std::array<u8, 256>;
}


/*  copy.hpp  */

namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR	

	void copy(image_soa const& src, image_t const& dst);

	void copy(image_t const& src, image_soa const& dst);

	void copy(image_soa const& src, view_t const& dst);

	void copy(view_t const& src, image_soa const& dst);

	void copy(image_soa const& src, image_soa const& dst);


#endif // !LIBIMAGE_NO_COLOR


	/*** copy parallel ***/

#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	

	void copy(image_t const& src, image_t const& dst);

	void copy(image_t const& src, view_t const& dst);

	void copy(view_t const& src, image_t const& dst);

	void copy(view_t const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);


	void copy(gray::image_t const& src, gray::image_t const& dst);

	void copy(gray::image_t const& src, gray::view_t const& dst);

	void copy(gray::view_t const& src, gray::image_t const& dst);

	void copy(gray::view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE

#endif // !LIBIMAGE_NO_PARALLEL


	/*** copy sequential ***/

	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR	

		void copy(image_t const& src, image_t const& dst);

		void copy(image_t const& src, view_t const& dst);

		void copy(view_t const& src, image_t const& dst);

		void copy(view_t const& src, view_t const& dst);

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		void copy(gray::image_t const& src, gray::image_t const& dst);

		void copy(gray::image_t const& src, gray::view_t const& dst);

		void copy(gray::view_t const& src, gray::image_t const& dst);

		void copy(gray::view_t const& src, gray::view_t const& dst);


#endif // !LIBIMAGE_NO_GRAYSCALE
	}
}


/*  transform.hpp  */

namespace libimage
{


	/*** transform parallel ***/

#ifndef LIBIMAGE_NO_PARALLEL

#ifndef LIBIMAGE_NO_COLOR	

	void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

	void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func);

	void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

	void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func);


	void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func);

	void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func);


	void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func);

	void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func);




#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	lookup_table_t to_lookup_table(u8_to_u8_f const& func);

	void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

	void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut);

	void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

	void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut);


	void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

	void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

	void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);


	void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut);

	void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut);


	void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func);

	void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

	void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

	void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

	void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR

#endif // !LIBIMAGE_NO_PARALLEL


	/*** transform sequential ***/

	namespace seq
	{
#ifndef LIBIMAGE_NO_COLOR

		void transform(image_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

		void transform(image_t const& src, view_t const& dst, pixel_to_pixel_f const& func);

		void transform(view_t const& src, image_t const& dst, pixel_to_pixel_f const& func);

		void transform(view_t const& src, view_t const& dst, pixel_to_pixel_f const& func);


		void transform_self(image_t const& src_dst, pixel_to_pixel_f const& func);

		void transform_self(view_t const& src_dst, pixel_to_pixel_f const& func);


		void transform_alpha(image_t const& src_dst, pixel_to_u8_f const& func);

		void transform_alpha(view_t const& src_dst, pixel_to_u8_f const& func);


#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

		void transform(gray::image_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

		void transform(gray::image_t const& src, gray::view_t const& dst, lookup_table_t const& lut);

		void transform(gray::view_t const& src, gray::image_t const& dst, lookup_table_t const& lut);

		void transform(gray::view_t const& src, gray::view_t const& dst, lookup_table_t const& lut);


		void transform(gray::image_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void transform(gray::image_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);

		void transform(gray::view_t const& src, gray::image_t const& dst, u8_to_u8_f const& func);

		void transform(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func);


		void transform_self(gray::image_t const& src_dst, lookup_table_t const& lut);

		void transform_self(gray::view_t const& src_dst, lookup_table_t const& lut);


		void transform_self(gray::image_t const& src_dst, u8_to_u8_f const& func);

		void transform_self(gray::view_t const& src_dst, u8_to_u8_f const& func);


#endif // !LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_COLOR
#ifndef LIBIMAGE_NO_GRAYSCALE

		void transform(image_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void transform(image_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

		void transform(view_t const& src, gray::image_t const& dst, pixel_to_u8_f const& func);

		void transform(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func);

#endif // !LIBIMAGE_NO_GRAYSCALE
#endif // !LIBIMAGE_NO_COLOR
	}
}


/*  alpha_blend.hpp  */

#ifndef LIBIMAGE_NO_COLOR

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

