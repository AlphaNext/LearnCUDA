all: gemm

# nvcc -o $@ $^ -arch=sm_52 -Xptxas -v -O3 -lcudart
gemm: cudaMallocPitch_and_cudaMemcpy2D.cu
	nvcc -o $@ $^ -arch=sm_52 -O3 -lcudart

clean:
	rm -f gemm *~
