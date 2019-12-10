all: gemm
#gemm: gemm_v1.cpp
#	nvcc -o $@ $^ -arch=sm_52 -Xptxas -v -O3 -lcublas
gemm: gemm_v0.cpp
	nvcc -o $@ $^ -arch=sm_52 -Xptxas -v -O3 -lcublas
clean:
	rm -f gemm
