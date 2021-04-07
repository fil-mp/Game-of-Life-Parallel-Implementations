#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>

#define HANDLE_ERROR( err ) ( HandleError( err, __FILE__, __LINE__ ) )
static void HandleError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define cudaCheckErrors(msg) \
	do { \
		cudaError_t __err = cudaGetLastError(); \
		if (__err != cudaSuccess) { \
			fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
				msg, cudaGetErrorString(__err), \
				__FILE__, __LINE__); \
			fprintf(stderr, "*** FAILED - ABORTING\n"); \
			return 1; \
		} \
	} while (0)

//for __syncthreads()
#ifndef __CUDACC__
#define __CUDACC__
#endif // !(__CUDACC__)
#include <device_functions.h>

#define ALIVE 1
#define DEAD 0

#define threads 32


__device__ int diff = 0;

__global__ void halo_rows(int dim, char* grid)
{
	// We want id ∈ [1,dim]
	int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

	if (id <= dim)
	{
		//Copy first real row to bottom ghost row
		grid[(dim + 2) * (dim + 1) + id] = grid[(dim + 2) + id];
		//Copy last real row to top ghost row
		grid[id] = grid[(dim + 2) * dim + id];
	}
}
__global__ void halo_columns(int dim, char* grid)
{
	// We want id ∈ [0,dim+1]
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if (id <= dim + 1)
	{
		//Copy first real column to right most ghost column
		grid[id * (dim + 2) + dim + 1] = grid[id * (dim + 2) + 1];
		//Copy last real column to left most ghost column
		grid[id * (dim + 2)] = grid[id * (dim + 2) + dim];
	}
}

__global__ void GOL(int dim, char* grid, char* gridAfter)
{
	int iy = (blockDim.y - 2) * blockIdx.y + threadIdx.y;
	int ix = (blockDim.x - 2) * blockIdx.x + threadIdx.x;

	int i = threadIdx.y;
	int j = threadIdx.x;


	// Declare the shared memory 
	__shared__ char gridBefore[threads][threads];

	
	if (ix <= dim + 1 && iy <= dim + 1)
		gridBefore[i][j] = grid[iy * (dim + 2) + ix];

	//Sync all threads in block
	__syncthreads();
	int sum = 0;
	if (iy <= dim && ix <= dim) {
		if (i != 0 && i != blockDim.y - 1 && j != 0 && j != blockDim.x - 1) {

			// Get the sum of neighbors 
			sum += gridBefore[i + 1][j]; //north
			sum += gridBefore[i - 1][j]; //south
			sum += gridBefore[i][j + 1];//east
			sum += gridBefore[i][j - 1]; //west
			sum += gridBefore[i + 1][j + 1];//northeast
			sum += gridBefore[i - 1][j - 1]; //northwest
			sum += gridBefore[i - 1][j + 1];//southeast
			sum += gridBefore[i + 1][j - 1];//southwest

			if (gridBefore[i][j] == ALIVE && sum < 2)
			{
				gridAfter[iy * (dim + 2) + ix] = DEAD;
				diff++;
			}
			else if (gridBefore[i][j] == ALIVE && (sum == 2 || sum == 3))
			{
				gridAfter[iy * (dim + 2) + ix] = ALIVE;
			}
			else if (gridBefore[i][j] == ALIVE && sum > 3)
			{
				gridAfter[iy * (dim + 2) + ix] = DEAD;
				diff++;
			}

			else if (gridBefore[i][j] == DEAD && sum == 3)
			{
				gridAfter[iy * (dim + 2) + ix] = ALIVE;
				diff++;
			}

			else {
				gridAfter[iy * (dim + 2) + ix] = gridBefore[i][j];
			}
		
		}
	}
}




int main(int argc, char* argv[])
{
	int i, j, iter, host_diff;
	char* h_grid;//Grid on host
	char* d_grid; //Grid on device
	char* d_gridAfter; //Second grid used on device only
	char* d_tmpGrid; //temporary grid pointer for swap

	int dim = 840; //Linear dimension of our grid - not counting ghost cells
	int generations = 1000; //Number of game steps

	// Allocate host Grid 
	h_grid = (char*)malloc(sizeof(char)*(dim + 2) * (dim + 2));
	
	// Allocate device grids
	cudaMalloc(&d_grid, sizeof(char)*(dim + 2) * (dim + 2));
	cudaCheckErrors("malloc failed");

	cudaMalloc(&d_gridAfter, sizeof(char)*(dim + 2) * (dim + 2));
	cudaCheckErrors("malloc failed");
	int psb;
	// initialize with possibilities
	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			psb = rand() % 100 + 1;
			if (psb <= 40)
				h_grid[i * (dim + 2) + j] = '1';
			else
				h_grid[i * (dim + 2) + j] = '0';
		}
	}

	cudaFuncSetCacheConfig(GOL, cudaFuncCachePreferShared);

	// Copy over initial game grid (Dim-1 threads)
	cudaMemcpy(d_grid, h_grid, sizeof(char)*(dim + 2) * (dim + 2), cudaMemcpyHostToDevice);
	dim3 blockSize(threads, threads, 1);
	int linGrid_x = (int)ceil(dim / (float)(threads - 2));
	int linGrid_y = (int)ceil(dim / (float)(threads - 2));
	dim3 gridSize(linGrid_x, linGrid_y, 1);

	dim3 cpyBlockSize(threads, 1, 1);
	dim3 cpyGridRowsGridSize((int)ceil(dim / (float)cpyBlockSize.x), 1, 1);
	dim3 cpyGridColsGridSize((int)ceil((dim + 2) / (float)cpyBlockSize.x), 1, 1);

	int counter = 0;
	float elapsed = 0;
	cudaEvent_t start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	HANDLE_ERROR(cudaEventRecord(start, 0));
	// Main game loop
	for (iter = 0; iter < generations; iter++) {
		host_diff = 0;
		cudaMemcpyToSymbol(&diff, &host_diff, sizeof(int), 0, cudaMemcpyHostToDevice);
		halo_rows << <cpyGridRowsGridSize, cpyBlockSize >> > (dim, d_grid);
		halo_columns << <cpyGridColsGridSize, cpyBlockSize >> > (dim, d_grid);
		GOL << <gridSize, blockSize >> > (dim, d_grid, d_gridAfter);
		cudaMemcpyFromSymbol(&host_diff, &diff, sizeof(int), 0, cudaMemcpyDeviceToHost);
		counter++;
		if (counter == 10)
		{
			if (host_diff == 0) {
				printf("No change or empty in %d\n", iter+1);
				break;
			}
			counter = 0;
		}
	
	// Swap
	d_tmpGrid = d_grid;
	d_grid = d_gridAfter;
	d_gridAfter = d_tmpGrid;
}
	
		

	// Copy back results and sum
	cudaMemcpy(h_grid, d_grid, sizeof(char)*(dim + 2) * (dim + 2), cudaMemcpyDeviceToHost);
	
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));

	printf("The elapsed time in gpu was %f\t sec\n", elapsed/1000);
	// Sum up alive cells and print results
	int total = 0;
	for (i = 1; i <= dim; i++) {
		for (j = 1; j <= dim; j++) {
			total += h_grid[i * (dim + 2) + j];
		}
	}
	printf("Total Alive: %d\n", total);

	cudaFree(d_grid);
	cudaFree(d_gridAfter);
	free(h_grid);
	

	return 0;
}

