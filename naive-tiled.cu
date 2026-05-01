#include <stdio.h>
#include <stdlib.h>

#define N 512
#define TILE_WIDTH 16

// ----------------------------------------
// Task 1: Naive Kernel
// Each thread computes one entry of C
// ----------------------------------------
__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            // Task 1: fill in the multiply-accumulate
	    sum = sum + A[row * n + k] * B[k * n + col];
        }
        // Task 1: write sum to C
	C[row * n + col] = sum;
    }
}

// ----------------------------------------
// Task 5: Tiled Kernel
// Each block computes one TILE_WIDTH x TILE_WIDTH tile of C
// ----------------------------------------
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < (n + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        int a_col = tile * TILE_WIDTH + threadIdx.x;
        int b_row = tile * TILE_WIDTH + threadIdx.y;

        // Task 5: load As (with boundary check)
	As[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;

        // Task 5: load Bs (with boundary check)
	Bs[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;
       
        // Task 6: Why do we need boundary checks when loading the tiles?
	/*
 		We need boundary checks when loading the tiles because our tile shape may not
		divide evenly with the dimension of matrix A or B. In the event that our tiles
		does not divide cleanly, then we have some entries in the tiles that are empty,
		which we set to zero. 			
	*/

	__syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            // Task 5: accumulate partial dot product
	    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();

	// Task 7: Why are the two __syncthreads() calls necessary?
	/*
		The two __syncthreads() calls are necessary since they ensure that the matrices A and B
		are fully loaded with data before performing the matrix computation. If we did not wait
		for all threads loading in the necessary data from A and B, and also, did not wait for
		the threads after performing the matrix computation, we run into the risk of incorrectly
		computing the output tile in C (since we computed with incomplete data), and also running
		into the risk of computing with overwritten data (since if we started performing computation
		for the next tile without having first finished our current tile, a thread from the next
		tile could overwrite data that we are performing for the current tile). 
	*/
    }

    if (row < n && col < n) {
        // Task 5: write sum to C
    }
}

int main() {
    int n = N;
    size_t bytes = n * n * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Task 2: set up block and grid dimensions
    dim3 blockDim(16, 16);  // fill in
    dim3 gridDim((n + blockDim.x - 1)/ blockDim.x,
    		 (n + blockDim.y - 1) / blockDim.y); 
    // Task 3: why does a naive kernel perform many redundant global memory accesses?
    /*
	A naive kernel performs many redundant global memory accesses because it needs to perform a memory access
	for each entry in matrix A and B to complete the inner product. It does not take advantage of the fact
	that memory accesses will load more bytes than one single float value, allowing us to perform memory
	accesses where all data in the access will be used for computation, instead of just retrieving one and 
	discarding the rest. In other words, the naive kernel does not coalesce memory. 

	Furthermore, the data that is retrieved from global memory is not shared between threads, meaning that each
	thread will need to perform a memory access for computation. This is redundant, since two vertical adjacent entries in
	C will require the same column from B, but each thread will fetch this column vector independently.  	

    */

    // Task 8: launch naive kernel
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    // Task 8: launch tiled kernel
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);    

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Task 9: verify correctness
    /*
	To verify correctness, we just check if each entry in the output matrix C matches to our expectations.
	Since we instantiated matrix A to have all entries of 1.0 and matrix B to have all entries of 2.0, each
	entry in C should be N * 1.0 * 2.0. If our matrix C does not match all entries to what we expect, then 
	our algorithm did not run correctly. 
    */
    // expected value per entry is N * 1.0 * 2.0 = 1024.0
    float expected = (float)n * 2.0f;
    int errors = 0;
    for (int i = 0; i < n * n; i++) {
	    errors = (h_C[i] != expected) ? errors + 1 : errors;
    }
    printf("Correctness: %s\n", errors == 0 ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}

