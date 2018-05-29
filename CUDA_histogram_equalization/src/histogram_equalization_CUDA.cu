/*
 ============================================================================
 Name        : histogram_equalization_CUDA.cu
 Author      : francesco
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void make_histogram(unsigned char *image, int width, int height, int *histogram){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;

	for(int i = idx; i < width * height; i += blockDim.x * gridDim.x){

		index = i * 3;

		int R = image[index];
		int G = image[index + 1];
		int B = image[index + 2];

		int Y = R * .299000 + G * .587000 + B * .114000;
		int U = R * -.168736 + G * -0.331264 + B * .500000 + 128;
		int V = R * .500000 + G * -.418688 + B * -.081312 + 128;

		atomicAdd(&(histogram[Y]),1);

		image[index] = Y;
		image[index + 1] = U;
		image[index + 2] = V;
	}

	__syncthreads();
}

__global__ void equalize(int *equalized, int *cumulative_dist, int *histogram, int width, int height){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for(int k = idx; k < 256; k += blockDim.x * gridDim.x){
		equalized[k] = (int)(((float)cumulative_dist[k] - histogram[0])/((float)width * height - 1) * 255);
	}
}

__global__ void YUV2RGB(unsigned char *image, int *cumulative_dist,int *histogram, int *equalized, int width, int height){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	long index;

	for(int i = idx; i < width * height; i += blockDim.x * gridDim.x){

		index = i * 3;

		int Y = equalized[image[index]];
		int U = image[index + 1];
		int V = image[index + 2];

		unsigned char R = (unsigned char)max(0, min(255,(int)(Y + 1.4075 * (V - 128))));
		unsigned char G = (unsigned char)max(0, min(255,(int)(Y - 1.3455 * (U - 128) - (.7169 * (V - 128)))));
		unsigned char B = (unsigned char)max(0, min(255,(int)(Y + 1.7790 * (U - 128))));

		image[index] = R;
		image[index + 1] = G;
		image[index + 2] = B;

	}

}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err){
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

int main(){

	Mat image = imread("src/images/desk.jpg");		//load the image

	if(!image.data){
		cout << "no image found";
		return -1;
	}

	int width = image.cols;
	int height = image.rows;

	int host_histogram[256];						//cpu histogram
	int host_equalized[256];						//cpu equalized histogram
	int host_cumulative_dist[256];

	unsigned char *host_image = image.ptr();		//Mat image to array image

	for(int i = 0; i < 256; i++){
		host_histogram[i] = 0;
	}

	unsigned char *device_image;	//gpu image

	int *device_histogram;			//gpu histogram
	int *device_equalized;			//gpu equalized histogram
	int *device_cumulative_dist;	//gpu cumulative dist.

	CUDA_CHECK_RETURN(cudaMalloc(&device_image, sizeof(char) * (width * height * 3)));										//gpu space allocation
	CUDA_CHECK_RETURN(cudaMalloc(&device_histogram, sizeof(int) * 256));													//
	CUDA_CHECK_RETURN(cudaMalloc(&device_equalized, sizeof(int) * 256));													//
	CUDA_CHECK_RETURN(cudaMalloc(&device_cumulative_dist, sizeof(int) * 256));												//

	CUDA_CHECK_RETURN(cudaMemcpy(device_image, host_image, sizeof(char) * (width * height * 3), cudaMemcpyHostToDevice));	//copy to gpu
	CUDA_CHECK_RETURN(cudaMemcpy(device_histogram, host_histogram, sizeof(int) * 256, cudaMemcpyHostToDevice));				//

	int block_size = 256;
	int grid_size = (width * height + (block_size - 1))/block_size;

	make_histogram<<<grid_size, block_size>>> (device_image, width, height, device_histogram);		//call first kernel

	CUDA_CHECK_RETURN(cudaMemcpy(host_histogram, device_histogram, sizeof(int) * 256, cudaMemcpyDeviceToHost));

	host_cumulative_dist[0] = host_histogram[0];										//compute cumulative dist. in cpu
																						//
	for(int i = 1; i < 256; i++){														//
		host_cumulative_dist[i] = host_histogram[i] + host_cumulative_dist[i-1];		//
	}																					//

	CUDA_CHECK_RETURN(cudaMemcpy(device_cumulative_dist, host_cumulative_dist, sizeof(int) * 256, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(device_equalized, host_equalized, sizeof(int) * 256, cudaMemcpyHostToDevice));

	equalize<<<grid_size, block_size>>>(device_equalized, device_cumulative_dist, device_histogram, width, height);					//call second kernel

	YUV2RGB<<<grid_size, block_size>>>(device_image, device_cumulative_dist, device_histogram, device_equalized, width, height);	//call third kernel

	CUDA_CHECK_RETURN(cudaMemcpy(host_image, device_image, sizeof(char) * (width * height * 3), cudaMemcpyDeviceToHost));

	CUDA_CHECK_RETURN(cudaFree(device_image));						//free gpu
	CUDA_CHECK_RETURN(cudaFree(device_histogram));					//
	CUDA_CHECK_RETURN(cudaFree(device_equalized));					//
	CUDA_CHECK_RETURN(cudaFree(device_cumulative_dist));			//

	cout << "correctly freed memory \n";

	Mat final_image = Mat(Size(width,height), CV_8UC3, host_image);
	imwrite("src/saved/desk.jpg", final_image);						//save equalized RGB image
	cout << "correctly saved image";

	return 0;

}
