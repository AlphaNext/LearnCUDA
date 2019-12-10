// http://www.orangeowlsolutions.com/archives/613
// https://github.com/OrangeOwlSolutions/General-CUDA-programming/wiki/cudaMallocPitch-and-cudaMemcpy2D

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<assert.h>

#define BLOCKSIZE_x 8
#define BLOCKSIZE_y 8
#define BLOCKSIZE_z 8
// x*y*z < 1024
// x*y*z < 1024
// x*y*z < 1024

#define Nrows 3
#define Ncols 5
#define NN 512
//#define NN 8
/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int hostPtr, int b){ return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/******************/
/* TEST KERNEL 2D */
/******************/
// https://stackoverflow.com/questions/16619274/cuda-griddim-and-blockdim/16619633
// https://stackoverflow.com/questions/16724844/divide-the-work-of-a-3d-kernel-among-blocks
__global__ void test_kernel_3D(float **devPtr, size_t pitch)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
    int tidz = blockIdx.z*blockDim.z + threadIdx.z;
 
//	printf("%i %i %i\n", tidx, tidy, tidz);
    if ((tidx < Ncols) && (tidy < Nrows) && (tidz < NN))
    {
		float *matrix_a = devPtr[tidz];
        float *row_a = (float *)((char*)matrix_a + tidy * pitch);
        row_a[tidx] = row_a[tidx] * 2.0;
    }
}

__global__ void test_kernel_2D(float *devPtr, size_t pitch)
{
    int tidx = blockIdx.x*blockDim.x + threadIdx.x;
    int tidy = blockIdx.y*blockDim.y + threadIdx.y;
 
    if ((tidx < Ncols) && (tidy < NN*Nrows))
    {
        float *row_a = (float *)((char*)devPtr+ tidy * pitch);
        row_a[tidx] = row_a[tidx] * 2.0;
    }
}

/********/
/* MAIN */
/********/
int main()
{
//    float hostPtr[Nrows][Ncols];
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
    float *devPtr;
    size_t pitch;
	gpuErrchk(cudaMallocPitch(&devPtr, &pitch, Ncols * sizeof(float), Nrows*NN));
    gpuErrchk(cudaMemcpy2D(devPtr, pitch, hostPtr, Ncols*sizeof(float), Ncols*sizeof(float), Nrows*NN, cudaMemcpyHostToDevice));
	
	float **ListPtr = 0;
	ListPtr = (float**)malloc(NN*sizeof(float*));

	for(int i=0; i<NN; i++){
		ListPtr[i] = devPtr + pitch/sizeof(float)*Nrows*i;
	}
	float **devListPtr = 0;
	gpuErrchk(cudaMalloc(&devListPtr, NN*sizeof(float*)));
	gpuErrchk(cudaMemcpy(devListPtr, ListPtr, NN*sizeof(float*), cudaMemcpyHostToDevice));
	// 3D 
    dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(Nrows, BLOCKSIZE_y), iDivUp(NN, BLOCKSIZE_z));
	dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);
	test_kernel_3D << <gridSize, blockSize >> >(devListPtr, pitch);
    // 2D
	//dim3 gridSize(iDivUp(Ncols, BLOCKSIZE_x), iDivUp(NN*Nrows, BLOCKSIZE_y));
	//dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y);
	//test_kernel_2D << <gridSize, blockSize >> >(devPtr, pitch);
	
    cudaPeekAtLastError();
	cudaDeviceSynchronize();
	
	//cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows*NN, cudaMemcpyDeviceToHost);
	gpuErrchk(cudaMemcpy2D(hostPtr, Ncols * sizeof(float), devPtr, pitch, Ncols * sizeof(float), Nrows*NN, cudaMemcpyDeviceToHost));
	for(int k=0; k<NN; k++){
		printf("-----iteration: %i-----\n", k);
		for (int i = 0; i < Nrows; i++){
	   		for (int j = 0; j < Ncols; j++)
				printf("%.1f ", hostPtr[k*Nrows*Ncols+i*Ncols+j]);
	      	//	printf("N %i row %i column %i value %f \n", k, i, j, hostPtr[k*Nrows*Ncols+i*Nrows+j]);
			printf("\n");
		}
		printf("-----done!-----\n");
	}
	return 0;
}


