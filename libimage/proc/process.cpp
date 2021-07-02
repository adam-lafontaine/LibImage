#include "process.hpp"
#include "convolve.hpp"

#include <algorithm>
#include <execution>
#include <cmath>
#include <array>


constexpr u32 VIEW_MIN_DIM = 5;

namespace libimage
{
	static u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
	{
		return static_cast<u8>(0.299 * red + 0.587 * green + 0.114 * blue);
	}


	static u8 pixel_grayscale_standard(pixel_t const& p)
	{
		return rgb_grayscale_standard(p.red, p.green, p.blue);
	}


	static u8 lerp_clamp(u8 src_low, u8 src_high, u8 dst_low, u8 dst_high, u8 val)
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


	static pixel_t alpha_blend_linear(pixel_t const& src, pixel_t const& current)
	{
		auto const a = static_cast<r32>(src.alpha) / 255.0f;

		auto const blend = [&](u8 s, u8 c)
		{
			auto sf = static_cast<r32>(s);
			auto cf = static_cast<r32>(c);

			auto blended = a * cf + (1.0f - a) * sf;

			return static_cast<u8>(blended);
		};

		auto red = blend(src.red, current.red);
		auto green = blend(src.green, current.green);
		auto blue = blend(src.blue, current.blue);

		return to_pixel(red, green, blue);
	}


	static bool verify(view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(gray::view_t const& view)
	{
		return view.image_data && view.width && view.height;
	}


	static bool verify(view_t const& src, view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify(view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static bool verify(gray::view_t const& src, gray::view_t const& dst)
	{
		return verify(src) && verify(dst) && dst.width == src.width && dst.height == src.height;
	}


	static r32 q_inv_sqrt(r32 n)
	{
		const float threehalfs = 1.5F;
		float y = n;

		long i = *(long*)&y;

		i = 0x5f3759df - (i >> 1);
		y = *(float*)&i;

		y = y * (threehalfs - ((n * 0.5F) * y * y));

		return y;
	}


	static r32 rms_contrast(gray::view_t const& view)
	{
		assert(verify(view));

		auto const norm = [](auto p) { return p / 255.0f; };

		auto total = std::accumulate(view.begin(), view.end(), 0.0f);
		auto mean = norm(total / (view.width * view.height));

		total = std::accumulate(view.begin(), view.end(), 0.0f, [&](r32 total, u8 p) { auto diff = norm(p) - mean; return diff * diff; });

		auto inv_mean = (view.width * view.height) / total;

		return q_inv_sqrt(inv_mean);
	}
	

	void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
	{
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
	{
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), dst.begin(), func);
	}


	void convert(gray::view_t const& src, u8_to_u8_f const& func)
	{
		assert(verify(src));

		auto const conv = [&](u8& p) { p = func(p); };
		std::for_each(src.begin(), src.end(), conv);
	}


	void copy(view_t const& src, view_t const& dst)
	{
		assert(verify(src, dst));

		std::copy(src.begin(), src.end(), dst.begin());
	}


	void copy(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		std::copy(src.begin(), src.end(), dst.begin());
	}


	void convert_grayscale(view_t const& src, gray::view_t const& dst)
	{
		convert(src, dst, pixel_grayscale_standard);
	}


	void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
	{
		assert(verify(view));

		auto const conv = [&](pixel_t& p) { p.alpha = func(p); };
		std::for_each(view.begin(), view.end(), conv);
	}


	void convert_alpha_grayscale(view_t const& view)
	{
		convert_alpha(view, pixel_grayscale_standard);
	}


	void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		convert(src, dst, conv);
	}


	void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high)
	{
		assert(src_low < src_high);

		u8 dst_low = 0;
		u8 dst_high = 255;

		auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
		convert(src, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		convert(src, dst, conv);
	}


	void binarize(gray::view_t const& src, u8 min_threashold)
	{
		auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
		convert(src, conv);
	}


	void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		convert(src, dst, conv);
	}


	void binarize(gray::view_t const& src, u8_to_bool_f const& func)
	{
		auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
		convert(src, conv);
	}


	void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
	{
		assert(verify(src, current));
		assert(verify(src, dst));

		std::transform(src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
	}


	void alpha_blend(view_t const& src, view_t const& current_dst)
	{
		assert(verify(src, current_dst));

		std::transform(src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
	}


	void blur(gray::view_t const& src, gray::view_t const& dst)
	{
		assert(verify(src, dst));

		u32 const width = src.width;
		u32 const height = src.height;

		assert(width >= VIEW_MIN_DIM);
		assert(height >= VIEW_MIN_DIM);

		// top and bottom rows equal to src
		u32 x_first = 0;
		u32 y_first = 0;
		u32 x_last = width - 1;
		u32 y_last = height - 1;
		auto src_top = src.row_begin(y_first);
		auto src_bottom = src.row_begin(y_last);
		auto dst_top = dst.row_begin(y_first);
		auto dst_bottom = dst.row_begin(y_last);
		for (u32 x = x_first; x <= x_last; ++x) // top and bottom rows
		{
			dst_top[x] = src_top[x];
			dst_bottom[x] = src_bottom[x];
		}

		// left and right columns equal to src
		x_first = 0;
		y_first = 1;
		x_last = width - 1;
		y_last = height - 2;
		for (u32 y = y_first; y <= y_last; ++y)
		{
			auto src_row = src.row_begin(y);
			auto dst_row = dst.row_begin(y);
			dst_row[x_first] = src_row[x_first];
			dst_row[x_last] = src_row[x_last];
		}

		// first inner top and bottom rows use 3 x 3 gaussian kernel
		x_first = 1;
		y_first = 1;
		x_last = width - 2;
		y_last = height - 2;		
		dst_top = dst.row_begin(y_first);
		dst_bottom = dst.row_begin(y_last);
		for (u32 x = x_first; x <= x_last; ++x)
		{
			dst_top[x] = gauss3(src, x, y_first);
			dst_bottom[x] = gauss3(src, x, y_last);
		}

		// first inner left and right columns use 3 x 3 gaussian kernel
		x_first = 1;
		y_first = 2;
		x_last = width - 2;
		y_last = height - 3;
		for (u32 y = y_first; y <= y_last; ++y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x_first] = gauss3(src, x_first, y);
			dst_row[x_last] = gauss3(src, x_last, y);
		}

		// inner pixels use 5 x 5 gaussian kernel
		x_first = 2;
		y_first = 2;
		x_last = width - 3;
		y_last = height - 3;		
		for (u32 y = y_first; y <= y_last; ++y)
		{
			auto dst_row = dst.row_begin(y);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				dst_row[x] = gauss5(src, x, y);
			}
		}
	}


	void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
	{
		assert(verify(src, dst));

		auto const width = src.width;
		auto const height = src.height;

		gray::image_t temp;
		auto temp_view = make_view(temp, src.width, src.height);
		blur(src, temp_view);

		// top and bottom rows are black
		u32 x_first = 0;
		u32 y_first = 0;
		u32 x_last = width - 1;
		u32 y_last = height - 1;
		auto dst_top = dst.row_begin(y_first);
		auto dst_bottom = dst.row_begin(y_last);
		for (u32 x = x_first; x <= x_last; ++x)
		{
			dst_top[x] = 0;
			dst_bottom[x] = 0;
		}

		// left and right columns are black
		x_first = 0;
		y_first = 1;
		x_last = width - 1;
		y_last = height - 2;
		for (u32 y = y_first; y <= y_last; ++y)
		{
			auto dst_row = dst.row_begin(y);
			dst_row[x_first] = 0;
			dst_row[x_last] = 0;
		}

		// get gradient magnitude of inner pixels
		x_first = 1;
		y_first = 1;
		x_last = width - 2;
		y_last = height - 2;
		for (u32 y = y_first; y <= y_last; ++y)
		{
			auto dst_row = dst.row_begin(y);
			for (u32 x = x_first; x <= x_last; ++x)
			{
				auto gx = x_gradient(temp_view, x, y);
				auto gy = y_gradient(temp_view, x, y);
				auto g = std::hypot(gx, gy);
				dst_row[x] = g < threshold ? 0 : 255;
			}
		}
	}


	namespace par
	{
		void convert(view_t const& src, gray::view_t const& dst, pixel_to_u8_f const& func)
		{
			assert(verify(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(gray::view_t const& src, gray::view_t const& dst, u8_to_u8_f const& func)
		{
			assert(verify(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), func);
		}


		void convert(gray::view_t const& src, u8_to_u8_f const& func)
		{
			assert(verify(src));

			auto const conv = [&](u8& p) { p = func(p); };
			std::for_each(std::execution::par, src.begin(), src.end(), conv);
		}


		void copy(view_t const& src, view_t const& dst)
		{
			assert(verify(src, dst));

			std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
		}


		void copy(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));

			std::copy(std::execution::par, src.begin(), src.end(), dst.begin());
		}


		void convert_grayscale(view_t const& src, gray::view_t const& dst)
		{
			par::convert(src, dst, pixel_grayscale_standard);
		}


		void convert_alpha(view_t const& view, pixel_to_u8_f const& func)
		{
			assert(verify(view));

			auto const update = [&](pixel_t& p) { p.alpha = func(p); };
			std::for_each(view.begin(), view.end(), update);
		}


		void convert_alpha_grayscale(view_t const& view)
		{
			par::convert_alpha(view, pixel_grayscale_standard);
		}


		void adjust_contrast(gray::view_t const& src, gray::view_t const& dst, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			par::convert(src, dst, conv);
		}


		void adjust_contrast(gray::view_t const& src, u8 src_low, u8 src_high)
		{
			assert(src_low < src_high);

			u8 dst_low = 0;
			u8 dst_high = 255;

			auto const conv = [&](u8 p) { return lerp_clamp(src_low, src_high, dst_low, dst_high, p); };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8 min_threashold)
		{
			auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
			par::convert(src, dst, conv);
		}


		void binarize(gray::view_t const& src, u8 min_threashold)
		{
			auto const conv = [&](u8 p) { return p >= min_threashold ? 255 : 0; };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& func)
		{
			auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
			par::convert(src, conv);
		}


		void binarize(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& func)
		{
			auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
			par::convert(src, dst, conv);
		}


		void binarize(gray::view_t const& src, u8_to_bool_f const& func)
		{
			auto const conv = [&](u8 p) { return func(p) ? 255 : 0; };
			par::convert(src, conv);
		}


		void alpha_blend(view_t const& src, view_t const& current, view_t const& dst)
		{
			assert(verify(src, current));
			assert(verify(src, dst));

			std::transform(std::execution::par, src.begin(), src.end(), current.begin(), dst.begin(), alpha_blend_linear);
		}


		void alpha_blend(view_t const& src, view_t const& current_dst)
		{
			assert(verify(src, current_dst));

			std::transform(std::execution::par, src.begin(), src.end(), current_dst.begin(), current_dst.begin(), alpha_blend_linear);
		}
				

		void blur(gray::view_t const& src, gray::view_t const& dst)
		{
			assert(verify(src, dst));
			auto const width = src.width;
			auto const height = src.height;

			assert(width >= VIEW_MIN_DIM);
			assert(height >= VIEW_MIN_DIM);

			// lamdas, lots of lamdas
			

			auto const copy_top = [&]() 
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = width;
				range.y_begin = 0;
				range.y_end = 1;
				auto src_top = sub_view(src, range);
				auto dst_top = sub_view(dst, range);

				par::copy(src_top, dst_top);
			};

			auto const copy_bottom = [&]() 
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = 1;
				range.y_begin = height - 1;
				range.y_end = height;
				auto src_bottom = sub_view(src, range);
				auto dst_bottom = sub_view(dst, range);

				par::copy(src_bottom, dst_bottom);
			};

			auto const copy_left = [&]() 
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = 1;
				range.y_begin = 1;
				range.y_end = height - 1;
				auto src_left = sub_view(src, range);
				auto dst_left = sub_view(dst, range);

				par::copy(src_left, dst_left);
			};

			auto const copy_right = [&]()
			{
				pixel_range_t range = {};
				range.x_begin = width - 1;
				range.x_end = width;
				range.y_begin = 1;
				range.y_end = height - 1;
				auto src_right = sub_view(src, range);
				auto dst_right = sub_view(dst, range);

				par::copy(src_right, dst_right);
			};

			// first inner top and bottom rows use 3 x 3 gaussian kernel
			auto const inner_gauss_top_bottom = [&]() 
			{
				u32 const x_begin = 1;
				u32 const x_length = width - 2 * x_begin;
				u32 const y_first = 1;
				u32 const y_last = height - 2;

				auto dst_top = dst.row_begin(y_first);
				auto dst_bottom = dst.row_begin(y_last);

				std::vector<u32> x_ids(x_length);
				std::iota(x_ids.begin(), x_ids.end(), x_begin); // TODO: ranges

				auto const gauss = [&](u32 x)
				{
					dst_top[x] = gauss3(src, x, y_first);
					dst_bottom[x] = gauss3(src, x, y_last);
				};

				std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss);
			};

			// first inner left and right columns use 3 x 3 gaussian kernel
			auto const inner_gauss_left_right = [&]() 
			{
				u32 const x_first = 1;
				u32 const x_last = width - 2;
				u32 const y_begin = 2;
				u32 const y_length = height - 2 * y_begin;

				std::vector<u32> y_ids(y_length);
				std::iota(y_ids.begin(), y_ids.end(), y_begin); // TODO: ranges

				auto const gauss = [&](u32 y) 
				{
					auto dst_row = dst.row_begin(y);
					dst_row[x_first] = gauss3(src, x_first, y);
					dst_row[x_last] = gauss3(src, x_last, y);
				};
				std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss);
			};

			// inner pixels use 5 x 5 gaussian kernel
			u32 const x_begin = 2;
			u32 const x_length = width - 2 * x_begin;
			u32 const y_begin = 2;
			u32 const y_length = height - 2 * y_begin;

			std::vector<u32> y_ids(y_length);
			std::iota(y_ids.begin(), y_ids.end(), y_begin); // TODO: ranges
			std::vector<u32> x_ids(x_length);
			std::iota(x_ids.begin(), x_ids.end(), x_begin); // TODO: ranges			

			auto const gauss_row = [&](u32 y) 
			{
				auto dst_row = dst.row_begin(y);

				auto const gauss_x = [&](u32 x)
				{
					dst_row[x] = gauss5(src, x, y);
				};

				std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), gauss_x);
			};

			auto const inner_gauss = [&]() 
			{
				std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), gauss_row);
			};

			// put the lambdas in an array
			std::array<std::function<void()>, 7> f_list =
			{
				copy_top,
				copy_bottom,
				copy_left,
				copy_right,
				inner_gauss_top_bottom,
				inner_gauss_left_right,
				inner_gauss
			};

			// finally execute everything
			std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
		}


		void edges(gray::view_t const& src, gray::view_t const& dst, u8 threshold)
		{
			assert(verify(src, dst));

			auto const width = src.width;
			auto const height = src.height;

			auto const zero = [](u8 p) { u8 val = 0;  return val; };

			auto const zero_top = [&]()
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = width;
				range.y_begin = 0;
				range.y_end = 1;
				auto dst_top = sub_view(dst, range);

				par::convert(dst_top, zero);
			};

			auto const zero_bottom = [&]()
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = 1;
				range.y_begin = height - 1;
				range.y_end = height;
				auto dst_bottom = sub_view(dst, range);

				par::convert(dst_bottom, zero);
			};

			auto const zero_left = [&]()
			{
				pixel_range_t range = {};
				range.x_begin = 0;
				range.x_end = 1;
				range.y_begin = 1;
				range.y_end = height - 1;
				auto dst_left = sub_view(dst, range);

				par::convert(dst_left, zero);
			};

			auto const zero_right = [&]()
			{
				pixel_range_t range = {};
				range.x_begin = width - 1;
				range.x_end = width;
				range.y_begin = 1;
				range.y_end = height - 1;
				auto dst_right = sub_view(dst, range);

				par::convert(dst_right, zero);
			};

			// get gradient magnitude of inner pixels
			u32 const x_begin = 1;
			u32 const x_length = width - 2 * x_begin;
			u32 const y_begin = 1;
			u32 const y_length = height - 2 * y_begin;

			std::vector<u32> y_ids(y_length);
			std::iota(y_ids.begin(), y_ids.end(), y_begin); // TODO: ranges
			std::vector<u32> x_ids(x_length);
			std::iota(x_ids.begin(), x_ids.end(), x_begin); // TODO: ranges

			gray::image_t temp;
			auto temp_view = make_view(temp, width, height);
			par::blur(src, temp_view);

			auto const grad_row = [&](u32 y) 
			{
				auto dst_row = dst.row_begin(y);

				auto const grad_x = [&](u32 x) 
				{
					auto gx = x_gradient(temp_view, x, y);
					auto gy = y_gradient(temp_view, x, y);
					auto g = std::hypot(gx, gy);
					dst_row[x] = g < threshold ? 0 : 255;
				};

				std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), grad_x);
			};

			auto const gradients_inner = [&]() 
			{
				std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), grad_row);
			};

			// put the lambdas in an array
			std::array<std::function<void()>, 5> f_list
			{
				zero_top,
				zero_bottom,
				zero_left,
				zero_right,
				gradients_inner
			};

			// finally execute everything
			std::for_each(std::execution::par, f_list.begin(), f_list.end(), [](auto const& f) { f(); });
		}

	}

}