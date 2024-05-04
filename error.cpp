#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <inttypes.h>
#include <omp.h>


double rel_accuracy(cv::Mat img1, cv::Mat img2){

    //Calculating ||Host|| l2 Norm
    long long int host_accum = 0;
    double host_l2;
    #pragma omp parallel for reduction (+:host_accum)
    for(int i=0;i<img1.rows;i++){
        for(int j=0;j<img1.cols;j++){
            cv::Vec3b intensity = img1.at<cv::Vec3b>(j, i);
            for(int k=0;k<3;i++){
                host_accum+=(int)intensity.val[k]*(int)intensity.val[k];
            }
        }
    }
    host_l2 = sqrt(host_accum);

    //Calculating ||Dev-Host|| l2
    double dev_sub_host_l2 = 0;
    #pragma omp parallel for reduction (+:dev_sub_host_l2)
    for(int i=0;i<img2.rows;i++){
        for(int j=0;j<img2.cols;j++){
            cv::Vec3b intensity1 = img1.at<cv::Vec3b>(j, i);
            cv::Vec3b intensity2 = img2.at<cv::Vec3b>(j, i);
            for(int k=0;k<3;i++){
                double sub_squared = (double)intensity2.val[k]-(double)intensity1.val[k];
                dev_sub_host_l2 += sub_squared*sub_squared;
            }
        }
    }
    dev_sub_host_l2  = sqrt(dev_sub_host_l2);
    return dev_sub_host_l2 / host_l2;
}

int main(int argc, char* argv[]){

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_COLOR);

    double acc = rel_accuracy(image1, image2);
    std::cout<<"Relative Error: "<<acc<<std::endl;

    return 0;
}