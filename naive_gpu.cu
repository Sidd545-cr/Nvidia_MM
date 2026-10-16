#include <cuda.h>
#include <stdio.h>
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



void print_array(int M, int N, float *arr){
    for(int i =0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f ",arr[j+N*i]);
        }
        printf("\n");
    }
}

void readMatrix(FILE *inputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%f", &matrix[i*cols+j]);
		}
	}
}

void writeMatrix(FILE *outputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%f ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}


__global__ void multiply(int M, int N, int L, float *arr1, float *arr2, float *sol){
    int x = blockDim.x*blockIdx.x+threadIdx.x;
    int y = blockDim.y*blockIdx.y+threadIdx.y;

    if(x<L && y<M){
        float sum =0.0f;
        for(int i=0;i<N;i++){
            sum += arr1[y*N+i]*arr2[L*i+x];
        }
        sol[y*L+x] = sum;
    }
}

int main(int argc, char **argv){
    int M,N,L;
    char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n");
		return 0;
	}

	fscanf(inputFilePtr, "%d %d %d", &M, &N, &L);
    // printf("[+] Enter M N L: ");
    // scanf("%d %d %d",&M,&N,&L);
    
    size_t s1,s2,s3;
    s1 = M*N*sizeof(float);
    s2 = L*N*sizeof(float);
    s3 = M*L*sizeof(float);
    float *arr1 = (float *)malloc(s1);
    float *arr2 = (float *)malloc(s2);
    float *sol = (float *)malloc(s3);
    float *darr1 = NULL;
    float *darr2 = NULL;
    float *dsol = NULL;

    readMatrix(inputFilePtr, M, N, arr1);
	readMatrix(inputFilePtr, N, L, arr2);
    print_array(M,N,arr1);
    print_array(N,L,arr2);

    CUDA_CHECK(cudaFree(0));
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, multiply));
    cudaEvent_t kstart, kstop;
    CUDA_CHECK(cudaEventCreate(&kstart));
    CUDA_CHECK(cudaEventCreate(&kstop));

    double e2e_start = now_ms();

    cudaMalloc(&darr1,s1);
    cudaMalloc(&darr2,s2);
    cudaMalloc(&dsol,s3);

    cudaMemcpy(darr1,arr1,s1,cudaMemcpyHostToDevice);
    cudaMemcpy(darr2,arr2,s2,cudaMemcpyHostToDevice);


    dim3 block(16, 16);
    dim3 grid(CEIL_DIV(L, 16), CEIL_DIV(M, 16));
    
    CUDA_CHECK(cudaEventRecord(kstart));
    multiply<<<grid, block>>>(M, N, L, darr1, darr2, dsol);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kstop));

    CUDA_CHECK(cudaMemcpy(sol,dsol,s3,cudaMemcpyDeviceToHost));
    double e2e_ms = now_ms() - e2e_start;

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(kstop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kstart, kstop));
    CUDA_CHECK(cudaEventDestroy(kstart));
    CUDA_CHECK(cudaEventDestroy(kstop));

    outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, M, L, sol);
    if(M<32&&L<32)
        print_array(M,L,sol);

    double flop = 2.0*M*N*L;
    printf("GPU kernel time (ms): %.3f  (%.2f GFLOP/s)\n", kernel_ms, flop/(kernel_ms*1e6));
    printf("GPU end-to-end time (ms): %.3f  (%.2f GFLOP/s)\n", e2e_ms, flop/(e2e_ms*1e6));

    cudaFree(darr1);
    cudaFree(darr2);
    cudaFree(dsol);
    free(arr1); 
    free(arr2);
    free(sol);
    fclose(inputFilePtr); 
    fclose(outputFilePtr);
    
}