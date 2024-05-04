#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iomanip> 
#include <cuda_runtime.h>
#include <cmath> 
#include <inttypes.h>
#include <iostream>
#include <fstream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


#define BLUR_SIZE 6 //Diameter
#define BLUR_RADIUS ((BLUR_SIZE-1)/2)
#define TILE_DIM 32
#define FILTER_SIZE 3//Diameter
#define FILTER_RADIUS ((FILTER_SIZE-1)/2)
#define HORIZONTAL 0
#define VERTICAL 1
#define CCLOCK_WISE 0
#define CLOCK_WISE 1

/*rgb2grey_kernel -- RGB to Greyscale Transformation
*   uint8_t* red        - pointer to array of image red channel
*   uint8_t* green      - pointer to array of image green channel
*   uint8_t* blue       - pointer to array of image blue channel
*   uint8_t* grey       - pointer to array for greyscale output image
*   unsigned int height - Height of input image
*   unsigned int wdith  - Height of input image
*/
__global__ void rgb2grey_kernel(uint8_t* red, uint8_t* green, uint8_t* blue, uint8_t* grey, unsigned int width, unsigned int height);

/*contrast_kernel -- Contrast Stretching
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *c_img      - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void contrast_kernel(uint8_t *img, uint8_t *c_img, unsigned int width, unsigned int height);

/*gauss_blur_kernel -- Gaussian Blur/Smoothing
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *blur_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void gauss_blur_kernel(uint8_t *img, uint8_t *blur_img, unsigned int width, unsigned int height);

/*flip_kernel -- Image Flip
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *flip_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*   unsigned int d_flag - Flip direction flag - HORIZONTAL or VERTICAL
*/
__global__ void flip_kernel(uint8_t* img, uint8_t* flip_img, unsigned int width, unsigned int height, unsigned int d_flag);

/*rotate_kernel -- Image Rotate
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *rot_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*   unsigned int d_flag - Rotate direction flag - CLOCK_WISE or CCLOCK_WISE
*/
__global__ void rotate_kernel(uint8_t* img, uint8_t* rot_img, unsigned int width, unsigned int height, unsigned int d_flag);

/*laplacian_filter_kernel -- Edge Detection - Laplacian
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *det_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void laplacian_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height);

/*sobel_filter_kernel -- Edge Detection - Sobel
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *det_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void sobel_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height);

/*naive_gauss_blur_kernel -- Gaussian Blur/Smoothing -- only uses shared memory for kernel
*   uint8_t *img        - pointer greyscale input image array of bytes - single channel
*   uint8_t *blur_img   - pointer for output image - single channel
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void naive_gauss_blur_kernel(uint8_t* img, uint8_t* blur_img, unsigned int width, unsigned int height);

/*load_image_bin -- Load Binary Image
*   const char* filename        - Name of binary image file
*   uint8_t *b_dest             - Pointer to blue channel destination buffer
*   uint8_t *g_dest             - Pointer to blue channel destination buffer
*   uint8_t *r_dest             - Pointer to blue channel destination buffer
*   unsigned int width          - Width of input image
*   unsigned int height         - Height of input image
*/
__host__ void load_image_bin(const char* filename, uint8_t *b_dest, uint8_t *g_dest, uint8_t *r_dest, unsigned int img_height, unsigned int img_width);

/*write_image_bin -- Write Image Binary
*   const char* filename    - Name of binary image file
*   uint8_t *source    - Pointer to output buffer - single channel - assumes you have sungle channel greyscale
*   unsigned int width  - Width of output image
*   unsigned int height - Height of output image
*/
__host__ void write_image_bin(const char* filename, uint8_t *source, unsigned int img_height, unsigned int img_width);
