// http://www.orangeowlsolutions.com/archives/613
// https://github.com/OrangeOwlSolutions/General-CUDA-programming/wiki/cudaMallocPitch-and-cudaMemcpy2D

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

//#include "Utilities.cuh"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16
#define BLOCKSIZE_z 16

#define Nrows 3
#define Ncols 5
#define NN 512
/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

/******************/
/* TEST KERNEL 2D */
/******************/
__global__ void test_kernel_2D(float **devPtr, size_t pitch)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tidz = blockIdx.z*blockDim.z + threadIdx.z;
 
	printf("%i %i %i\n", tidx, tidy, tidz);
    if ((tidx < Ncols) && (tidy < Nrows) && (tidz < NN))
    {
		float *matrix_a = devPtr[tidz];
		printf("%i %i %i\n", tidx, tidy, tidz);
        float *row_a = (float *)((char*)matrix_a + tidy * pitch);
        row_a[tidx] = row_a[tidx] * 2.0;
		if(tidz == 0){
			printf("####%i####\n", tidz);
			printf("%.1f ", row_a[tidx]);
			printf("####%i####\n", tidz);
		}
    }
}

/********/
/* MAIN */
/********/
int main()
{
//    float hostPtr[Nrows][Ncols];
    float *devPtr;
    size_t pitch;
 	float xx = 0.0;
	float *hostPtr = (float*)malloc(NN*Nrows*Ncols*sizeof(float));
	for(int k=0; k<NN; k++){
    	for (int i = 0; i < Nrows; i++){
    		for (int j = 0; j < Ncols; j++) {
    			hostPtr[k*Nrows*Ncols+i*Ncols+j] = xx++;;
    			//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
 			}
		}
	}
	// --- 2D pitched allocation and host->device memcopy
	cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows*NN);
	cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows*NN, cudaMemcpyHostToDevice);
	
	float **ListPtr = 0;
	ListPtr = (float**)malloc(NN*sizeof(float*));

	for(int i=0; i<NN; i++){
		ListPtr[i] = devPtr + pitch/sizeof(float)*Nrows*i;
	}
	float **devListPtr = 0;
	cudaMalloc(&devListPtr, NN*sizeof(float*));
	cudaMemcpy(devListPtr, ListPtr, NN*sizeof(float*), cudaMemcpyHostToDevice);
	dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y), iDivUp(NN, BLOCKSIZE_z));
	dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);
	
	cudaError_t status = test_kernel_2D << <gridSize, blockSize >> >(devListPtr, pitch);
	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	
	//cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows*NN, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows*NN, cudaMemcpyDeviceToHost);
	
//	for(int k=0; k<NN; k++){
//		printf("-----iteration: %i-----\n", k);
//		for (int i = 0; i < Nrows; i++){
//	   		for (int j = 0; j < Ncols; j++)
//				printf("%.1f ", hostPtr[k*Nrows*Ncols+i*Ncols+j]);
//	      	//	printf("N %i row %i column %i value %f \n", k, i, j, hostPtr[k*Nrows*Ncols+i*Nrows+j]);
//			printf("\n");
//		}
//		printf("-----done!-----\n");
//	}
	return 0;
}


//
///******************/
///* TEST KERNEL 2D */
///******************/
//__global__ void test_kernel_2D(float *devPtr, size_t pitch)
//{
//	int    tidx = blockIdx.x*blockDim.x + threadIdx.x;
//	int    tidy = blockIdx.y*blockDim.y + threadIdx.y;
//	
//	if ((tidx < Ncols) && (tidy < Nrows))
//	{
//		float *row_a = (float *)((char*)devPtr + tidy * pitch);
//		row_a[tidx] = row_a[tidx] * tidx * tidy;
//	}
//}
//
///********/
///* MAIN */
///********/
//int main()
//{
//	float hostPtr[Nrows][Ncols];
//	float *devPtr;
//	size_t pitch;
//
//	for (int i = 0; i < Nrows; i++)
//		for (int j = 0; j < Ncols; j++) {
//			hostPtr[i][j] = 1.f;
//			//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
//		}
//
//	// --- 2D pitched allocation and host->device memcopy
//	cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows);
//	cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows, cudaMemcpyHostToDevice);
//
//	dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y));
//	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
//
//	test_kernel_2D << <gridSize, blockSize >> >(devPtr, pitch);
//	cudaPeekAtLastError();
//	cudaDeviceSynchronize();
//
//	cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows, cudaMemcpyDeviceToHost);
//
//	for (int i = 0; i < Nrows; i++) 
//		for (int j = 0; j < Ncols; j++) 
//			printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
//
//	return 0;
//
//}
