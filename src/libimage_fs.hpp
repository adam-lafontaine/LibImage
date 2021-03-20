#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "libimage.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace libimage
{
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
}