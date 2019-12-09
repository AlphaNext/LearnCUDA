all: gemm

gemm: cudaMallocPitch_and_cudaMemcpy2D.cu
	nvcc -o $@ $^ -arch=sm_60 -v -O3 -lcudart

clean:
	rm -f gemm *~
