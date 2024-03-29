#pragma once

#include "../defines.hpp"

#define STBI_NO_JPEG

#ifndef LIBIMAGE_PNG
#define STBI_NO_PNG
#endif // !LIBIMAGE_PNG

#ifndef LIBIMAGE_BMP
#define STBI_NO_BMP
#endif // !LIBIMAGE_BMP


#define STBI_NO_GIF
#define STBI_NO_PSD
#define STBI_NO_PIC
#define STBI_NO_PNM
#define STBI_NO_HDR
#define STBI_NO_TGA


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#ifndef LIBIMAGE_NO_WRITE
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif // !LIBIMAGE_NO_WRITE


#ifndef LIBIMAGE_NO_RESIZE
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#endif // !LIBIMAGE_NO_RESIZE
