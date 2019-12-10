// https://github.com/pyrovski/cublasSgemmBatched-example/blob/master/gemm.cpp
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
using namespace std;

int main(int argc, char ** argv){
    
    int status;
    int lower = 2;
    int upper = 100;
    int num = 128;
    int reps = 5;
    int verbose = 0;
    
    // mxk && k*n
    int m, k, n;
    int NN = 128;
    num = NN;
    m = 6, k = 5, n = 4;
    float *matricesA = (float*)malloc(m * k * num * sizeof(float));
    float *matricesB = (float*)malloc(k * n * num * sizeof(float));
    float *matricesC = (float*)malloc(m * n * num * sizeof(float));
    
    assert(matricesA);
    assert(matricesB);
    assert(matricesC);
    int ind =11;
    int i, j, x;
    // define matrix a
    for(x=0; x<NN; x++){
        ind = 11;
        for(j=0; j<k; j++){
            for(i=0; i<m; i++){
                matricesA[x*m*k+j*m+i] = (float)ind++;
            }
        }
    }
    
    // define matrix b
    for(x=0; x<NN; x++){
        ind = 11;
        for(j=0; j<n; j++){
            for(i=0; i<k; i++){
                matricesB[x*k*n+j*k+i] = (float)ind++;
            }
        }
    }
    
    for(x=0; x<NN; x++){
        for(j=0; j<n; j++){
            for(i=0; i<m; i++){
                matricesC[x*m*n+j*m+i] = 0.0f;
            }
        }
    }
    
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    
    stat = cublasCreate(&handle);
    if(stat != CUBLAS_STATUS_SUCCESS){
        cerr << "cublas init failed" << endl;
        exit(1);
    }
    
    if(verbose) cout << "allocating device variables" << endl;
    
    // allocate input space on device
    float *devMatricesA, *devMatricesB, *devMatricesC;
    size_t devMatricesPitchA, devMatricesPitchB, devMatricesPitchC;
    cudaStat =
    cudaMallocPitch(&devMatricesA,
                    &devMatricesPitchA,
                    m * sizeof(float),
                    num * k);
    assert(!cudaStat);
    cudaStat =
    cudaMallocPitch(&devMatricesB,
                    &devMatricesPitchB,
                    k * sizeof(float),
                    num * n);
    assert(!cudaStat);
    // allocate result space on device
    cudaStat =
    cudaMallocPitch(&devMatricesC,
                    &devMatricesPitchC,
                    m * sizeof(float),
                    num * n);
    assert(!cudaStat);
    
    if(verbose) cout << "copying data to device" << endl;
    // copy data to device
    cudaStat =
    cudaMemcpy2D(devMatricesA,
                 devMatricesPitchA,
                 matricesA,
                 m * sizeof(float),
                 m * sizeof(float),
                 k * num,
                 cudaMemcpyHostToDevice);
    assert(!cudaStat);
    
    cudaStat =
    cudaMemcpy2D(devMatricesB,
                 devMatricesPitchB,
                 matricesB,
                 k * sizeof(float),
                 k * sizeof(float),
                 n * num,
                 cudaMemcpyHostToDevice);
    assert(!cudaStat);
    
    // create lists of device pointers to inputs and outputs
    float **AList = 0, **BList = 0, **CList = 0;
    
    AList = (float**)malloc(num * sizeof(float*));
    BList = (float**)malloc(num * sizeof(float*));
    CList = (float**)malloc(num * sizeof(float*));

    for(int i = 0; i < num; i++){
        AList[i] = devMatricesA + devMatricesPitchA/sizeof(float) * k * i;
        BList[i] = devMatricesB + devMatricesPitchB/sizeof(float) * n * i;
        CList[i] = devMatricesC + devMatricesPitchC/sizeof(float) * n * i;
    }
    
    // copy pointer lists to device
    float **devAList = 0, **devBList = 0, **devCList = 0;
    cudaStat = cudaMalloc(&devAList, num * sizeof(float*));
    assert(!cudaStat);
    
    cudaStat = cudaMalloc(&devBList, num * sizeof(float*));
    assert(!cudaStat);
    
    cudaStat = cudaMalloc(&devCList, num * sizeof(float*));
    assert(!cudaStat);
    
    cudaStat = cudaMemcpy(devAList,
                          AList,
                          num * sizeof(float*),
                          cudaMemcpyHostToDevice);
    assert(!cudaStat);
    
    cudaStat = cudaMemcpy(devBList,
                          BList,
                          num * sizeof(float*),
                          cudaMemcpyHostToDevice);
    assert(!cudaStat);
    
    cudaStat = cudaMemcpy(devCList,
                          CList,
                          num * sizeof(float*),
                          cudaMemcpyHostToDevice);
    assert(!cudaStat);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int
    lda = devMatricesPitchA / sizeof(float),
    ldb = devMatricesPitchB / sizeof(float),
    ldc = devMatricesPitchC / sizeof(float);
    //  int lda = m, ldb = k, ldc = m;
    const float alpha = 1.0f, beta = 0.0f;
    double sum = 0.0;
    for(int rep = 0; rep < reps; rep++){
        cudaEventRecord(start, 0);
        stat = cublasSgemmBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  m,
                                  n,
                                  k,
                                  &alpha,
                                  (const float**)devAList,
                                  lda,
                                  (const float**)devBList,
                                  ldb,
                                  &beta,
                                  devCList,
                                  ldc,
                                  num);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        if(stat != CUBLAS_STATUS_SUCCESS){
            cerr << "cublasSgemmBatched failed" << endl;
            exit(1);
        }
        assert(!cudaGetLastError());
        
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        elapsed /= 1000.0f;
        sum += elapsed;
    }
    cout << " average: " << sum/reps << " s; " << sum / reps / num << " s per operation" << endl;
    // copy result to host
    for(i=0; i<NN; i++){
        cudaStat = cudaMemcpy2D(matricesC+m*n*i, m*sizeof(float),
                                CList[i], devMatricesPitchC,
                                m*sizeof(float), n, cudaMemcpyDeviceToHost);
        assert(!cudaStat);
    }
    printf("Matrix A :\n");
    for(i=0; i<m; i++){
        for(j=0; j<k; j++){
            printf("%7.0f ", matricesA[IDX2C(i,j,m)]);      // print c after Sgemm
        }
        printf(" \n");
    }
    printf("Matrix B :\n");
    for(i=0; i<k; i++){
        for(j=0; j<n; j++){
            printf("%7.0f ", matricesB[k*n+IDX2C(i,j,k)]);      // print c after Sgemm
        }
        printf(" \n");
    }
    
    printf("c after Sgemm :\n");
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            printf("%7.0f ", matricesC[IDX2C(i,j,m)]);      // print c after Sgemm
        }
        printf(" \n");
    }
    printf("check other iteration: \n");
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            printf("%7.0f ", matricesC[m*n+IDX2C(i,j,m)]);      // print c after Sgemm
        }
        printf(" \n");
    }

    cudaFree(devAList);                   // free device memory
    cudaFree(devBList);                   // free device memory
    cudaFree(devCList);                   // free device memory
    cudaFree(devMatricesA);
    cudaFree(devMatricesB);
    cudaFree(devMatricesC);
    free(matricesA);
    free(matricesB);
    free(matricesC);
    cublasDestroy(handle);           // destroy CUBLAS context
    return 0;
}
