#include "img_lib.cuh"


__global__ void rgb2grey_kernel(uint8_t* red, uint8_t* green, uint8_t* blue, uint8_t* grey, unsigned int width, unsigned int height){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i=((row*width)+col);i<width*height;i+=(gridDim.x*blockDim.x)*(gridDim.y*blockDim.y)){
        grey[i] = (uint8_t)(blue[i]*.10) + (uint8_t)(green[i]*.6) + (uint8_t)(red[i]*.3);
    }
}

__global__ void contrast_kernel(uint8_t *img, uint8_t *c_img, unsigned int width, unsigned int height)
{

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int bid = (threadIdx.y * blockDim.x) + threadIdx.x;

    __shared__ uint8_t min;
    __shared__ uint8_t max;
   if(bid == 0){
       min = 0;
   }
   else if(bid == 1){
       max = 255;
   }
   __syncthreads();

    for(int i=((row*width)+col);i<width*height;i+=(gridDim.x*blockDim.x)*(gridDim.y*blockDim.y)){
        c_img[i] = (uint8_t)(255 * ((double)(img[i]-min)/(double)(max-min)));
    }
}

__global__ void gauss_blur_kernel(uint8_t *img, uint8_t *blur_img, unsigned int width, unsigned int height)
{
    __shared__ float shared_kernel[BLUR_SIZE*BLUR_SIZE];
    extern __shared__  uint8_t shared_input[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_width = blockDim.x + 2*BLUR_RADIUS;
    int shared_height = blockDim.y + 2*BLUR_RADIUS;

    __shared__ float sum;
    if(threadIdx.x == 0 && threadIdx.y == 0){
        sum = 0;
    }
     __syncthreads();

    //Calculating 2D gaussian filter and storing it in shared memory
    if(row < height  && col < width){
        if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
            double x = threadIdx.y - ((float)BLUR_SIZE - 1) / 2.0;
            double y = threadIdx.x - ((float)BLUR_SIZE - 1) / 2.0;
            shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x] = 1 * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(1, 2)))) * (-1));   
        }
    }
    __syncthreads();

    //Atomically adding each value to sum for nomalization
    if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
        atomicAdd(&sum,shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x]);
    }
    __syncthreads();

    //Normalizing the kernel
    if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
        shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x] /= sum;
    }

    //Loading shared input
    for (int i = threadIdx.y; i < shared_height; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < shared_width; j += blockDim.x)
        {
            int in_row = blockIdx.y * blockDim.y + i - BLUR_RADIUS;
            int in_col = blockIdx.x * blockDim.x + j - BLUR_RADIUS;

            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
            {
                shared_input[i * shared_width + j] = img[in_row * width + in_col];
            }
            else
            {
                shared_input[i * shared_width + j] = 0;
            }
        }
    }
    __syncthreads();
    
    //Applying kernel to each pixel
    if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
    {
        float sum = 0;
        
        for (int i = -BLUR_RADIUS; i <= BLUR_RADIUS; i++)
        {
            for (int j = -BLUR_RADIUS; j <= BLUR_RADIUS; j++)
            {
                int shared_row = threadIdx.y + BLUR_RADIUS + i;
                int shared_col = threadIdx.x + BLUR_RADIUS + j;
                int kernel_row = i + BLUR_RADIUS;
                int kernel_col = j + BLUR_RADIUS;
                sum += (float)shared_input[shared_row * shared_width + shared_col] * shared_kernel[kernel_row * (2 * BLUR_RADIUS + 1) + kernel_col];
            }
        }
        
        if (row < height && col < width)
        {
            blur_img[row * width + col] = (uint8_t)(__float2int_rn(sum));

        }
    }
}


__global__ void flip_kernel(uint8_t* img, uint8_t* flip_img, unsigned int width, unsigned int height, unsigned int d_flag){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < height  && col < width){
        if(d_flag == HORIZONTAL){
            flip_img[(row*width)+(width-col)] = img[(row*width)+col];
        }
        else if (d_flag == VERTICAL)
        {
            flip_img[((height-row)*width)+col] = img[(row*width)+col];
        }
    }
}

__global__ void rotate_kernel(uint8_t* img, uint8_t* rot_img, unsigned int width, unsigned int height, unsigned int d_flag){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < height  && col < width){
        if(d_flag == CCLOCK_WISE){
            rot_img[((width-col)*height)+row] = img[(row*width)+col];
        }
        else if (d_flag == CLOCK_WISE)
        {
            rot_img[((col)*height)+(height-row)] = img[(row*width)+col];
        }
    }
}

__global__ void laplacian_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height)
{
    __shared__ int16_t shared_kernel1[FILTER_SIZE*FILTER_SIZE];
    extern __shared__  uint8_t shared_input[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_width = blockDim.x + 2*FILTER_RADIUS;
    int shared_height = blockDim.y + 2*FILTER_RADIUS;


    //Loading shared kernel
    if(row < height  && col < width){
        int16_t k[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
        if(threadIdx.y < FILTER_SIZE && threadIdx.x < FILTER_SIZE){
            shared_kernel1[threadIdx.y*FILTER_SIZE+threadIdx.x] = k[threadIdx.y*FILTER_SIZE+threadIdx.x];
        }
    }
   
    
    
    //Loading shared input
    for (int i = threadIdx.y; i < shared_height; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < shared_width; j += blockDim.x)
        {
            int in_row = blockIdx.y * blockDim.y + i - (FILTER_RADIUS);
            int in_col = blockIdx.x * blockDim.x + j - (FILTER_RADIUS);

            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
            {
                shared_input[i * shared_width + j] = img[in_row * width + in_col];
            }
            else
            {
                shared_input[i * shared_width + j] = 0;
            }
        }
    }
    __syncthreads();


    //Applying kernel to each pixel
    if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
    {
        int16_t sum = 0;
        
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++)
        {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++)
            {
                int shared_row = threadIdx.y + FILTER_RADIUS + i;
                int shared_col = threadIdx.x + FILTER_RADIUS + j;
                int kernel_row = i + FILTER_RADIUS;
                int kernel_col = j + FILTER_RADIUS;
                sum += ((int16_t)shared_input[shared_row * shared_width + shared_col] * shared_kernel1[kernel_row * (2 * FILTER_RADIUS + 1) + kernel_col]);
            }
        }
        
        if (row < height && col < width)
        {   
            sum = abs(sum);
            if(sum>255){sum = 255;}
            det_img[row * width + col] = (uint8_t)(sum);
        }
    }

}

__global__ void sobel_filter_kernel(uint8_t *img, uint8_t *det_img,unsigned int width, unsigned int height)
{
    __shared__ int16_t shared_kernel1[FILTER_SIZE*FILTER_SIZE];
    __shared__ int16_t shared_kernel2[FILTER_SIZE*FILTER_SIZE];
    extern __shared__  uint8_t shared_input[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int shared_width = blockDim.x + 2*FILTER_RADIUS;
    int shared_height = blockDim.y + 2*FILTER_RADIUS;

    

    //Loading shared kernel
    if(row < height  && col < width){
        int16_t k1[9] = {-1,0,1,-2,0,2,-1,0,1};
        int16_t k2[9] = {1,2,1,0,0,0,-1,-2,-1};
        if(threadIdx.y < FILTER_SIZE && threadIdx.x < FILTER_SIZE){
            shared_kernel1[threadIdx.y*FILTER_SIZE+threadIdx.x] = k1[threadIdx.y*FILTER_SIZE+threadIdx.x];
            shared_kernel2[threadIdx.y*FILTER_SIZE+threadIdx.x] = k2[threadIdx.y*FILTER_SIZE+threadIdx.x];
        }
    }

    //Loading shared input
    for (int i = threadIdx.y; i < shared_height; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < shared_width; j += blockDim.x)
        {
            int in_row = blockIdx.y * blockDim.y + i - FILTER_RADIUS;
            int in_col = blockIdx.x * blockDim.x + j - FILTER_RADIUS;

            if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width)
            {
                shared_input[i * shared_width + j] = img[in_row * width + in_col];
            }
            else
            {
                shared_input[i * shared_width + j] = 0;
            }
        }
    }
    __syncthreads();
    
    //Applying kernel to each pixel
    if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x)
    {
        int16_t sum1 = 0;
        int16_t sum2 = 0;
        
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++)
        {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++)
            {
                int shared_row = threadIdx.y + FILTER_RADIUS + i;
                int shared_col = threadIdx.x + FILTER_RADIUS + j;
                int kernel_row = i + FILTER_RADIUS;
                int kernel_col = j + FILTER_RADIUS;
                sum1 += ((int16_t)shared_input[shared_row * shared_width + shared_col] * shared_kernel1[kernel_row * (2 * FILTER_RADIUS + 1) + kernel_col]);
                sum2 += ((int16_t)shared_input[shared_row * shared_width + shared_col] * shared_kernel2[kernel_row * (2 * FILTER_RADIUS + 1) + kernel_col]);
            }
        }
        
        if (row < height && col < width)
        {   
            uint16_t total = abs(sum1)+abs(sum2);
            if(total > 255){total = 255;}//Clipping
            det_img[row * width + col] = (uint8_t)total;
        }
    }

}

__global__ void naive_gauss_blur_kernel(uint8_t* img, uint8_t* blur_img, unsigned int width, unsigned int height){
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared_kernel[BLUR_SIZE*BLUR_SIZE];

    __shared__ float sum;
    if(threadIdx.x == 0 && threadIdx.y == 0){
        sum = 0;
    }
     __syncthreads();

    //Calculating 2D gaussian filter and storing it in shared memory
    if(row < height  && col < width){
        if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
            double x = threadIdx.y - ((float)BLUR_SIZE - 1) / 2.0;
            double y = threadIdx.x - ((float)BLUR_SIZE - 1) / 2.0;
            shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x] = 1 * exp(((pow(x, 2) + pow(y, 2)) / ((2 * pow(1, 2)))) * (-1));   
        }
    }
    __syncthreads();

    //Atomically adding each value to sum for nomalization
    if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
        atomicAdd(&sum,shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x]);
    }
    __syncthreads();

    //Normalizing the kernel
    if(threadIdx.y < BLUR_SIZE && threadIdx.x < BLUR_SIZE){
        shared_kernel[threadIdx.y*BLUR_SIZE + threadIdx.x] /= sum;
    }

    if(row < height  && col < width){
        int sum = 0;
        for(int in_row = row-BLUR_SIZE; in_row<row+BLUR_SIZE+1;in_row++){
            for(int in_col = col-BLUR_SIZE; in_col<col+BLUR_SIZE+1;in_col++){
                if(in_row >=0 && in_row < height && in_col >= 0 && in_col <width){
                    sum += img[(in_row*width)+in_col]*shared_kernel[((in_row*width)%BLUR_SIZE)+(in_col%BLUR_SIZE)];    
                }
            }
        }
        blur_img[(row*width)+col] = (uint8_t)(sum/((2*BLUR_SIZE+1)*(2*BLUR_SIZE+1)));
    }
}

__host__ void load_image_bin(const char* filename, uint8_t *b_dest, uint8_t *g_dest, uint8_t *r_dest, unsigned int img_height, unsigned int img_width){
    //Opening binary file containing image
    FILE *file = fopen(filename, "rb");
    
    //Sorting binary by bgr values and storing them in the corresponding buffer
    for(int i=0;i<img_height;i++){
        for(int j=0;j<img_width;j++){
            for(int k=0;k<3;k++){
                char buf;
                switch(k)
                {
                case 0:
                    fread(&buf, 1, 1, file);
                    b_dest[(i*img_width)+j] = (uint8_t)buf;
                case 1:
                    fread(&buf, 1, 1, file); //Just to increment file pointer
                    g_dest[(i*img_width)+j] = (uint8_t)buf;
                case 2:
                    fread(&buf, 1, 1, file); //Just to increment file pointer
                    r_dest[(i*img_width)+j] = (uint8_t)buf;
                default:
                    NULL;
                }
                
            }
        }
    }
    //Closing binary file containing image
    fclose(file);
}

__host__ void write_image_bin(const char* filename, uint8_t *source, unsigned int img_height, unsigned int img_width){
    std::ofstream file_out;
    file_out.open(filename, std::ios::binary | std::ios::out);
    for (int i = 0; i < ((img_height*img_width)); i++)
    {
        for (int j = 0; j < 2; j++)
        {
            file_out << source[i] << source[i] << source[i];
        }
    }
    file_out.close();
}
