#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iomanip> 
#include <cuda_runtime.h>
#include <cmath> 
#include <inttypes.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)


#define BLUR_SIZE 6 //Diameter
#define BLUR_RADIUS ((BLUR_SIZE-1)/2)
#define TILE_DIM 32
#define FILTER_SIZE 3//Diameter
#define FILTER_RADIUS ((BLUR_SIZE-1)/2)
#define HORIZONTAL 0
#define VERTICAL 1
#define CCLOCK_WISE 0
#define CLOCK_WISE 1


/*contrast_kernel -- Contrast Stretching
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *c_img      - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void contrast_kernel(uint8_t *img, uint8_t *c_img, unsigned int width, unsigned int height);

/*gauss_blur_kernel -- Gaussian Blur/Smoothing
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *blur_img   - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void gauss_blur_kernel(uint8_t *img, uint8_t *blur_img, unsigned int width, unsigned int height);

/*flip_kernel -- Image Flip
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *flip_img   - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*   unsigned int d_flag - Flip direction flag - HORIZONTAL or VERTICAL
*/
__global__ void flip_kernel(uint8_t* img, uint8_t* flip_img, unsigned int width, unsigned int height, unsigned int d_flag);

/*rotate_kernel -- Image Rotate
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *rot_img   - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*   unsigned int d_flag - Rotate direction flag - CLOCK_WISE or CCLOCK_WISE
*/
__global__ void rotate_kernel(uint8_t* img, uint8_t* rot_img, unsigned int width, unsigned int height, unsigned int d_flag);

/*laplacian_filter_kernel -- Edge Detection - Laplacian
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *det_img   - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void laplacian_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height);

/*sobel_filter_kernel -- Edge Detection - Sobel
*   uint8_t *img        - pointer greyscale input image array of bytes
*   uint8_t *det_img   - pointer for output image
*   unsigned int width  - Width of input image
*   unsigned int height - Height of input image
*/
__global__ void sobel_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height);