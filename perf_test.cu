#include "img_lib.cuh"
#include <time.h>

using namespace std; 


__global__ void gen_bytes(uint8_t *input, unsigned int size){
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;  
    int tid = index_y * grid_width + index_x;
    if(tid == 0){
        for(int i=0;i<size;i++){
            uint8_t v = i%256;
            input[i] = v;
        }
    }
  __syncthreads(); 
}



int main(int argc, const char * argv[]){


    int runs_n = std::stoi(argv[1]);
    if(argc < 2 || argc > 2 || runs_n < 1){
        printf("Invalid number of runs per size_n\n");
        exit(0);
    }

    int size_n = 100;

    std::ofstream file_out;
    file_out.open("avg_times.txt", std::ios::out | std::ios::app);

    
    while(size_n < 6500){

        int num_runs = 0;
        double contrast_time = 0;
        double naive_time = 0;
        double gauss_time = 0;
        double flip_v_time = 0;
        double flip_h_time = 0;
        double rotate_c_time = 0;
        double rotate_cc_time = 0;
        double sobel_time = 0;
        double laplacian_time = 0;

        uint8_t *in_dev, *out_dev;
        CUDA_CALL(cudaMalloc(&in_dev, sizeof(uint8_t)*(size_n*size_n)));
        CUDA_CALL(cudaMalloc(&out_dev, sizeof(uint8_t)*(size_n*size_n)));
        

        dim3 numThreadsPerBlock(32,32);
        dim3 numBlocks((size_n+numThreadsPerBlock.x-1)/numThreadsPerBlock.x, (size_n+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);

        gen_bytes<<<numBlocks, numThreadsPerBlock>>>(in_dev, size_n*size_n);
        
        float gpu_ms = 0;
        cudaEvent_t start,stop;
        CUDA_CALL(cudaEventCreate(&start));
        CUDA_CALL(cudaEventCreate(&stop));

        int blur_tile_size = (32 + ((BLUR_SIZE-1)/2)) * (32 + ((BLUR_SIZE-1)/2)) * sizeof(uint8_t);
        int filter_tile_size = (32 + ((FILTER_SIZE-1)/2)) * (32 + ((FILTER_SIZE-1)/2)) * sizeof(uint8_t);

        while(num_runs < runs_n){
            cudaEvent_t start,stop;
            CUDA_CALL(cudaEventCreate(&start));
            CUDA_CALL(cudaEventCreate(&stop));
            CUDA_CALL(cudaEventRecord(start));
            contrast_kernel<<<numBlocks, numThreadsPerBlock>>>(in_dev, out_dev, size_n, size_n);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            contrast_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            gauss_blur_kernel<<<numBlocks, numThreadsPerBlock, blur_tile_size>>>(in_dev, out_dev, size_n, size_n);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            gauss_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            flip_kernel<<<numBlocks, numThreadsPerBlock>>>(in_dev, out_dev, size_n, size_n, HORIZONTAL);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            flip_h_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            flip_kernel<<<numBlocks, numThreadsPerBlock>>>(in_dev, out_dev, size_n, size_n, VERTICAL);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            flip_v_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            rotate_kernel<<<numBlocks, numThreadsPerBlock, blur_tile_size>>>(in_dev, out_dev, size_n, size_n, CLOCK_WISE);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            rotate_c_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            rotate_kernel<<<numBlocks, numThreadsPerBlock, blur_tile_size>>>(in_dev, out_dev, size_n, size_n, CCLOCK_WISE);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            rotate_cc_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            sobel_filter_kernel<<<numBlocks, numThreadsPerBlock, filter_tile_size>>>(in_dev, out_dev, size_n, size_n);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            sobel_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            laplacian_filter_kernel<<<numBlocks, numThreadsPerBlock, filter_tile_size>>>(in_dev, out_dev, size_n, size_n);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            laplacian_time += gpu_ms;

            CUDA_CALL(cudaEventRecord(start));
            naive_gauss_blur_kernel<<<numBlocks, numThreadsPerBlock>>>(in_dev, out_dev, size_n, size_n);
            CUDA_CALL(cudaEventRecord(stop));
            CUDA_CALL(cudaEventSynchronize(stop));
            gpu_ms = 0;
            CUDA_CALL(cudaEventElapsedTime(&gpu_ms, start, stop));
            naive_time += gpu_ms;

            CUDA_CALL(cudaEventDestroy(start));
            CUDA_CALL(cudaEventDestroy(stop));

            num_runs+=1;
        }

        printf("----------\n");
        printf("Image Size: %d x %d\n", size_n, size_n);
        printf("Contrast Stretch Avg Time: %lfms\n", (double)(contrast_time/num_runs));
        printf("Naive Gaussian Blur Avg Time: %lfms\n", (double)(naive_time/num_runs));
        printf("Gaussian Blur Avg Time: %lfms\n", (double)(gauss_time/num_runs));
        printf("Flip Horizontal Avg Time: %lfms\n", (double)(flip_h_time/runs_n));
        printf("Flip Vertical Avg Time: %lfms\n", (double)(flip_v_time/runs_n));
        printf("Rotate Clockwise Avg Time: %lfms\n", (double)(rotate_c_time/runs_n));
        printf("Rotate Counter-Clockwise Avg Time: %lfms\n", (double)(rotate_cc_time/runs_n));
        printf("Sobel Edge Detection Avg Time: %lfms\n", (double)(sobel_time/runs_n));
        printf("Laplacian Edge Detection Avg Time: %lfms\n", (double)(sobel_time/runs_n));
        printf("----------\n");



        file_out<<(double)(contrast_time/runs_n)<<","<<(double)(naive_time/runs_n)<<","<<(double)(gauss_time/runs_n)<<","<<(double)(flip_h_time/runs_n)<<","<<(double)(flip_v_time/runs_n)<<
        (double)(rotate_c_time/runs_n)<<","<<(double)(rotate_cc_time/runs_n)<<","<<(double)(sobel_time/runs_n)<<","<<(double)(laplacian_time/runs_n)<<std::endl;


        cudaFree(in_dev);
        cudaFree(out_dev);
        size_n*=2;
    }
    file_out.close();
    printf("Done!\n");
    
    return 0;
}
