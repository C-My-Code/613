#include "img_lib.cuh"


int main(int argc, const char *argv[]){

    
    std::cout<<"Image Name?:"<<std::endl;
    std::string image_name;
    std::cin>>image_name;
    std::cin.clear();

    //Using image_to_bin.py to transform into binary
    std::string in_script_location = "image_to_bin.py ";
    std::string py = "python3 ";
    std::string command =py+=in_script_location+=image_name;
    system(command.c_str());

    std::string height;
    std::cout<<"Image Height?:"<<std::endl;
    std::cin>>height;
    std::cin.clear();

    std::string width;
    std::cout<<"Image Width?:"<<std::endl;
    std::cin>>width;
    std::cin.clear();


    int uheight = std::stoi(height);
    uint uwidth = std::stoi(width);
    int size = uheight*uwidth;
    
  

    //Allocating host memory
    uint8_t *host_b = (uint8_t *)malloc(sizeof(uint8_t)*(size));
    uint8_t *host_g = (uint8_t *)malloc(sizeof(uint8_t)*(size));
    uint8_t *host_r = (uint8_t *)malloc(sizeof(uint8_t)*(size));
    uint8_t *host_grey = (uint8_t *)malloc(sizeof(uint8_t)*(size));

    
    load_image_bin("image.bin", host_b, host_g, host_r, uheight, uwidth);
    

    uint8_t *dev_b, *dev_g, *dev_r, *dev_grey, *dev_out1, *dev_out2;
    CUDA_CALL(cudaMalloc(&dev_b, sizeof(uint8_t)*(size)));
    CUDA_CALL(cudaMalloc(&dev_g, sizeof(uint8_t)*(size)));
    CUDA_CALL(cudaMalloc(&dev_r, sizeof(uint8_t)*(size)));
    CUDA_CALL(cudaMalloc(&dev_grey, sizeof(uint8_t)*(size)));

    CUDA_CALL(cudaMemcpy(dev_b, host_b, sizeof(uint8_t)*(size), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_g, host_g, sizeof(uint8_t)*(size), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_r, host_r, sizeof(uint8_t)*(size), cudaMemcpyHostToDevice));
    
    free(host_b);
    free(host_g);
    free(host_r);
    
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((uwidth+numThreadsPerBlock.x-1)/numThreadsPerBlock.x, (uheight+numThreadsPerBlock.y-1)/numThreadsPerBlock.y);

    int blur_tile_size = (32 + ((BLUR_SIZE-1)/2)) * (32 + ((BLUR_SIZE-1)/2)) * sizeof(uint8_t);
    int filter_tile_size = (32 + ((FILTER_SIZE-1)/2)) * (32 + ((FILTER_SIZE-1)/2)) * sizeof(uint8_t);
  
    rgb2grey_kernel<<<numBlocks, numThreadsPerBlock>>>(dev_r, dev_g, dev_b, dev_grey, uwidth, uheight);

    CUDA_CALL(cudaFree(dev_b));
    CUDA_CALL(cudaFree(dev_g));
    CUDA_CALL(cudaFree(dev_r));

    CUDA_CALL(cudaMalloc(&dev_out1, sizeof(uint8_t)*(size)));
    CUDA_CALL(cudaMalloc(&dev_out2, sizeof(uint8_t)*(size)));

    contrast_kernel<<<numBlocks, numThreadsPerBlock>>>(dev_grey, dev_out1, uwidth, uheight);
    CUDA_CALL(cudaFree(dev_grey));

    gauss_blur_kernel<<<numBlocks, numThreadsPerBlock, blur_tile_size>>>(dev_out1, dev_out2, uwidth, uheight);

    flip_kernel<<<numBlocks, numThreadsPerBlock>>>(dev_out2, dev_out1, uwidth, uheight, HORIZONTAL);

    rotate_kernel<<<numBlocks, numThreadsPerBlock>>>(dev_out1, dev_out2, uwidth, uheight, CCLOCK_WISE);

    laplacian_filter_kernel<<<numBlocks, numThreadsPerBlock, filter_tile_size>>>(dev_out2, dev_out1, uheight, uwidth);//Notice height and width switched places due to rotate

    CUDA_CALL(cudaMemcpy(host_grey, dev_out1, sizeof(uint8_t)*(size), cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(dev_out1));
    CUDA_CALL(cudaFree(dev_out2));


    write_image_bin("image_out.bin", host_grey, uwidth, uheight);//Notice height and width switched places due to rotate

    free(host_grey);

    
    std::string out_script_location = "bin_to_image.py ";
    py = "python3 ";
    std::string space = " ";
    command = py+=out_script_location+=width+=space+=height;//Notice height and width switched places due to rotate
    system(command.c_str());
    

    return 0; 
}