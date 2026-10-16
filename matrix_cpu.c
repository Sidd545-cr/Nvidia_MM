#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

double now_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int readMatrix(FILE *inputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) 
			fscanf(inputFilePtr, "%f", &matrix[(size_t)i*cols+j]);
	}
	return 0;
}

void writeMatrix(FILE *outputFilePtr, int rows, int cols, float *matrix) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%f ", matrix[(size_t)i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

void print_array(int M, int N, float *arr){
    for(int i =0;i<M;i++){
        for(int j=0;j<N;j++){
            printf("%f ",arr[(size_t)i*N+j]);
        }
        printf("\n");
    }
}

void multiply(int M, int N, int L, const float *arr1, const float *arr2, float *sol){
    memset(sol, 0, (size_t)M*L*sizeof(float));
    for(int i=0;i<M;i++){
        for(int k=0;k<N;k++){
            float a = arr1[(size_t)i*N+k];
            for(int j=0;j<L;j++){
                sol[(size_t)i*L+j] += a*arr2[(size_t)k*L+j];
            }
        }
    }
}

int main(int argc, char **argv){
     
    char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	FILE *inputFilePtr = fopen(inputFileName, "r");
	
    int M,N,L;
	fscanf(inputFilePtr, "%d %d %d", &M, &N, &L);
	    

    float *arr1 = malloc((size_t)M*N*sizeof(float));
    float *arr2 = malloc((size_t)N*L*sizeof(float));
    float *sol  = malloc((size_t)M*L*sizeof(float));

    readMatrix(inputFilePtr, M, N, arr1);
    readMatrix(inputFilePtr, N, L, arr2);
    fclose(inputFilePtr);


    double t0 = now_ms();
    multiply(M,N,L,arr1,arr2,sol);
    double cpu_ms = now_ms() - t0;

    FILE *outputFilePtr = fopen(outputFileName, "w");
    
	writeMatrix(outputFilePtr, M, L, sol);
	fclose(outputFilePtr);

    printf("CPU time (ms): %.3f  (%.2f GFLOP/s)\n", cpu_ms, 2.0*M*N*L/(cpu_ms*1e6));

    free(arr1); free(arr2); free(sol);
    return 0;
}
