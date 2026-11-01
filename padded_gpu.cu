#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); } } while (0)


void readMatrix(FILE *inputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%f", &matrix[(size_t)i*cols+j]);
		}
	}
}

void writeMatrix(FILE *outputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%f ", matrix[(size_t)i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}


__global__ void multiply_padded_16(int M, int N, int L, const float * __restrict__ arr1,
                                   const float * __restrict__ arr2, float * __restrict__ sol){
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*16 + ty;
    int col = blockIdx.x*16 + tx;
    float sum = 0.0f;
    for(int p=0; p<CEIL_DIV(N, 16); p++){
        int ac = p*16 + tx, br = p*16 + ty;
        As[ty][tx] = (row < M && ac < N) ? arr1[row*N + ac] : 0.0f;
        Bs[ty][tx] = (br < N && col < L) ? arr2[br*L + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for(int k=0; k<16; k++){
            sum += As[ty][k]*Bs[k][tx];
        }
        __syncthreads();
    }
    if(row < M && col < L){
        sol[row*L + col] = sum;
    }
}

__global__ void multiply_padded_32(int M, int N, int L, const float * __restrict__ arr1,
                                   const float * __restrict__ arr2, float * __restrict__ sol){
    __shared__ float As[32][33];
    __shared__ float Bs[32][33];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y*32 + ty;
    int col = blockIdx.x*32 + tx;
    float sum = 0.0f;
    for(int p=0; p<CEIL_DIV(N, 32); p++){
        int ac = p*32 + tx, br = p*32 + ty;
        As[ty][tx] = (row < M && ac < N) ? arr1[row*N + ac] : 0.0f;
        Bs[ty][tx] = (br < N && col < L) ? arr2[br*L + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for(int k=0; k<32; k++){
            sum += As[ty][k]*Bs[k][tx];
        }
        __syncthreads();
    }
    if(row < M && col < L){
        sol[row*L + col] = sum;
    }
}


typedef void (*tiled_kernel)(int, int, int, const float *, const float *, float *);


int main(int argc, char **argv){

    char *inputFileName = argv[1];
	char *outputFileName = argv[2];
    int tile = argc > 3 ? atoi(argv[3]) : 16;

    tiled_kernel kernel;
    switch(tile){
        case 16: kernel = multiply_padded_16; break;
        case 32: kernel = multiply_padded_32; break;
        default:
            fprintf(stderr, "Unsupported tile size '%s': use 16 or 32\n", argv[3]);
            return 1;
    }

	FILE *inputFilePtr = fopen(inputFileName, "r");

    int M,N,L;
	fscanf(inputFilePtr, "%d %d %d", &M, &N, &L);

    size_t s1 = (size_t)M*N*sizeof(float);
    size_t s2 = (size_t)N*L*sizeof(float);
    size_t s3 = (size_t)M*L*sizeof(float);
    float *arr1 = (float *)malloc(s1);
    float *arr2 = (float *)malloc(s2);
    float *sol  = (float *)malloc(s3);

    float *darr1 = NULL, *darr2 = NULL, *dsol = NULL;

    readMatrix(inputFilePtr, M, N, arr1);
	readMatrix(inputFilePtr, N, L, arr2);
    fclose(inputFilePtr);

    CUDA_CHECK(cudaFree(0));
    struct cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, (const void *)kernel));
    cudaEvent_t kstart, kstop;
    CUDA_CHECK(cudaEventCreate(&kstart));
    CUDA_CHECK(cudaEventCreate(&kstop));

    double e2e_start = now_ms();

    CUDA_CHECK(cudaMalloc((void **)&darr1, s1));
    CUDA_CHECK(cudaMalloc((void **)&darr2, s2));
    CUDA_CHECK(cudaMalloc((void **)&dsol, s3));

    CUDA_CHECK(cudaMemcpy(darr1, arr1, s1, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(darr2, arr2, s2, cudaMemcpyHostToDevice));

    dim3 block(tile, tile);
    dim3 grid(CEIL_DIV(L, tile), CEIL_DIV(M, tile));

    CUDA_CHECK(cudaEventRecord(kstart, 0));
    kernel<<<grid, block>>>(M, N, L, darr1, darr2, dsol);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kstop, 0));

    CUDA_CHECK(cudaMemcpy(sol, dsol, s3, cudaMemcpyDeviceToHost));
    double e2e_ms = now_ms() - e2e_start;

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(kstop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kstart, kstop));
    CUDA_CHECK(cudaEventDestroy(kstart));
    CUDA_CHECK(cudaEventDestroy(kstop));

    FILE *outputFilePtr = fopen(outputFileName, "w");

	writeMatrix(outputFilePtr, M, L, sol);
    fclose(outputFilePtr);

    double flop = 2.0*M*N*L;

    printf("GPU kernel time (ms): %.3f  (%.2f GFLOP/s)\n", kernel_ms, flop/(kernel_ms*1e6));
    printf("GPU end-to-end time (ms): %.3f  (%.2f GFLOP/s)\n", e2e_ms, flop/(e2e_ms*1e6));

    CUDA_CHECK(cudaFree(darr1));
    CUDA_CHECK(cudaFree(darr2));
    CUDA_CHECK(cudaFree(dsol));
    free(arr1);
    free(arr2);
    free(sol);
    return 0;
}
