#ifndef LIBIMAGE_NO_SIMD

#include "../verify.hpp"
#include "simd_def.hpp"


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
	1.0f, 0.0f, -1.0f,
	2.0f, 0.0f, -2.0f,
	1.0f, 0.0f, -1.0f,
};
constexpr std::array<r32, 9> GRAD_Y_3X3
{
	1.0f,  2.0f,  1.0f,
	0.0f,  0.0f,  0.0f,
	-1.0f, -2.0f, -1.0f,
};


namespace libimage
{
	namespace simd
	{
		


		static void gauss3_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			auto const do_simd = [&](int i)
			{
				MemoryVector mem{};
				u32 w = 0;
				auto acc_vec = simd_setzero();
				auto src_vec = simd_setzero();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;
						auto ptr = src_begin + offset;
						cast_copy_len(ptr, mem.data);

						src_vec = simd_load(mem.data);

						auto weight = simd_load_broadcast(GAUSS_3X3.data() + w);

						acc_vec = simd_fmadd(weight, src_vec, acc_vec);
					}
				}

				simd_store(mem.data, acc_vec);

				cast_copy_len(mem.data, dst_begin + i);
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class SRC_GRAY_IMG_T, class DST_GRAY_IMG_T>
		static void gauss_inner_top_bottom(SRC_GRAY_IMG_T const& src, DST_GRAY_IMG_T const& dst)
		{
			u32 x_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_begin = 1;
			u32 y_last = src.height - 2;

			auto length = x_end - x_begin;
			auto pitch = (u32)(src.row_begin(1) - src.row_begin(0));

			auto src_row = src.row_begin(y_begin) + x_begin;
			auto dst_row = dst.row_begin(y_begin) + x_begin;
			gauss3_row(src_row, dst_row, length, pitch);

			src_row = src.row_begin(y_last) + x_begin;
			dst_row = dst.row_begin(y_last) + x_begin;
			gauss3_row(src_row, dst_row, length, pitch);
		}


		static void gauss5_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			auto const do_simd = [&](int i)
			{
				MemoryVector mem{};
				u32 w = 0;
				auto acc_vec = simd_setzero();
				auto src_vec = simd_setzero();

				for (int ry = -2; ry < 3; ++ry)
				{
					for (int rx = -2; rx < 3; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;
						auto ptr = src_begin + offset;
						cast_copy_len(ptr, mem.data);

						src_vec = simd_load(mem.data);

						auto weight = simd_load_broadcast(GAUSS_5X5.data() + w);

						acc_vec = simd_fmadd(weight, src_vec, acc_vec);
					}
				}

				simd_store(mem.data, acc_vec);

				cast_copy_len(mem.data, dst_begin + i);
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void gauss5_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_begin = 2;
			u32 y_begin = 2;
			u32 x_end = src.width - 2;
			u32 y_end = src.height - 2;

			auto length = x_end - x_begin;
			auto pitch = (u32)(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				auto src_row = src.row_begin(y) + x_begin;
				auto dst_row = dst.row_begin(y) + x_begin;
				gauss5_row(src_row, dst_row, length, pitch);
			}
		}


		static void xy_gradients_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			auto const do_simd = [&](int i)
			{
				MemoryVector mem{};
				u32 w = 0;
				auto vec_x = simd_setzero();
				auto vec_y = simd_setzero();
				auto src_vec = simd_setzero();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;
						auto ptr = src_begin + offset;
						cast_copy_len(ptr, mem.data);

						src_vec = simd_load(mem.data);

						auto weight_x = simd_load_broadcast(GRAD_X_3X3.data() + w);
						auto weight_y = simd_load_broadcast(GRAD_Y_3X3.data() + w);

						vec_x = simd_fmadd(weight_x, src_vec, vec_x);
						vec_y = simd_fmadd(weight_y, src_vec, vec_y);
					}
				}

				vec_x = simd_multiply(vec_x, vec_x);
				vec_y = simd_multiply(vec_y, vec_y);

				auto grad = simd_sqrt(simd_add(vec_x, vec_y));
				simd_store(mem.data, grad);

				cast_copy_len(mem.data, dst_begin + i);
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void xy_gradients_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst)
		{
			u32 x_begin = 1;
			u32 y_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_end = src.height - 1;

			auto length = x_end - x_begin;
			auto pitch = (u32)(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				xy_gradients_row(src.row_begin(y) + x_begin, dst.row_begin(y) + x_begin, length, pitch);
			}
		}


		static void edges_row(u8* src_begin, u8* dst_begin, u32 length, u32 pitch, u8_to_bool_f const& cond)
		{
			constexpr u32 N = VEC_LEN;
			constexpr u32 STEP = N;

			auto const do_simd = [&](int i)
			{
				MemoryVector mem{};
				u32 w = 0;
				auto vec_x = simd_setzero();
				auto vec_y = simd_setzero();
				auto src_vec = simd_setzero();

				for (int ry = -1; ry < 2; ++ry)
				{
					for (int rx = -1; rx < 2; ++rx, ++w)
					{
						int offset = ry * pitch + rx + i;
						auto ptr = src_begin + offset;
						cast_copy_len(ptr, mem.data);

						src_vec = simd_load(mem.data);

						auto weight_x = simd_load_broadcast(GRAD_X_3X3.data() + w);
						auto weight_y = simd_load_broadcast(GRAD_Y_3X3.data() + w);

						vec_x = simd_fmadd(weight_x, src_vec, vec_x);
						vec_y = simd_fmadd(weight_y, src_vec, vec_y);
					}
				}

				vec_x = simd_multiply(vec_x, vec_x);
				vec_y = simd_multiply(vec_y, vec_y);

				auto grad = simd_sqrt(simd_add(vec_x, vec_y));
				simd_store(mem.data, grad);

				transform_len(mem.data, dst_begin + i, [&](r32 val) { return (u8)(cond((u8)val) ? 255 : 0); });
			};

			for (u32 i = 0; i < length - STEP; i += STEP)
			{
				do_simd(i);
			}

			do_simd(length - STEP);
		}


		template<class GRAY_SRC_IMG_T, class GRAY_DST_IMG_T>
		static void edges_inner(GRAY_SRC_IMG_T const& src, GRAY_DST_IMG_T const& dst, u8_to_bool_f const& cond)
		{
			u32 x_begin = 1;
			u32 y_begin = 1;
			u32 x_end = src.width - 1;
			u32 y_end = src.height - 1;

			auto length = x_end - x_begin;
			auto pitch = (u32)(src.row_begin(1) - src.row_begin(0));

			for (u32 y = y_begin; y < y_end; ++y)
			{
				edges_row(src.row_begin(y) + x_begin, dst.row_begin(y) + x_begin, length, pitch, cond);
			}
		}


		void inner_gauss(gray::image_t const& src, gray::image_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::image_t const& src, gray::view_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::view_t const& src, gray::image_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gauss(gray::view_t const& src, gray::view_t const& dst)
		{
			simd::gauss5_inner(src, dst);
		}


		void inner_gradients(gray::image_t const& src, gray::image_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::image_t const& src, gray::view_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::view_t const& src, gray::image_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_gradients(gray::view_t const& src, gray::view_t const& dst)
		{
			simd::xy_gradients_inner(src, dst);
		}


		void inner_edges(gray::image_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::image_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::view_t const& src, gray::image_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}


		void inner_edges(gray::view_t const& src, gray::view_t const& dst, u8_to_bool_f const& cond)
		{
			simd::edges_inner(src, dst, cond);
		}

	}
}

#endif // !LIBIMAGE_NO_SIMD