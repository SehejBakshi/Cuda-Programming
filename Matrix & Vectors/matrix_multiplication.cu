#include <bits/stdc++.h>
using namespace std;

__global__ void matrixMultiplication(int *A, int *B, int *C, int N);

void mm(int *A, int *B, int *C, int N);

int main()
{
	cudaEvent_t start, end;
	int ROWS = 2;
	int COLS = 2;

	int *hostA = (int *)malloc(sizeof(int) * ROWS * COLS);
	int *hostB = (int *)malloc(sizeof(int) * ROWS * COLS);
	int *hostC = (int *)malloc(sizeof(int) * ROWS * COLS);

	//initialize matrices A and B by random numbers
	srand(time(0));
	int i, j;
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			hostB[i * COLS + j] = rand() % 30;
			hostA[i * COLS + j] = rand() % 20;
		}
	}

	cout << "Matrix A: " << endl;
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			cout << hostA[i * COLS + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Matrix B: " << endl;
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			cout << hostB[i * COLS + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	int *deviceA, *deviceB, *deviceC;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	cudaMalloc(&deviceA, sizeof(int) * ROWS * COLS);
	cudaMalloc(&deviceB, sizeof(int) * ROWS * COLS);
	cudaMalloc(&deviceC, sizeof(int) * ROWS * COLS);

	cudaMemcpy(deviceA, hostA, sizeof(int) * ROWS * COLS, cudaMemcpyHostToDevice);

	cudaMemcpy(deviceB, hostB, sizeof(int) * ROWS * COLS, cudaMemcpyHostToDevice);

	mm(deviceA, deviceB, deviceC, ROWS);

	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		printf("Cuda failure %s: ", cudaGetErrorString(e));
	}

	cudaDeviceSynchronize();

	cudaMemcpy(hostC, deviceC, ROWS * COLS * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	//now do actual multiplication
	int N = ROWS;
	int *actual = (int *)malloc(sizeof(int) * ROWS * COLS);
	int sum;
	for (int row = 0; row < ROWS; row++)
	{
		for (int col = 0; col < COLS; col++)
		{
			sum = 0;
			for (int n = 0; n < N; n++)
			{
				sum += hostA[row * N + n] * hostB[n * N + col];
			}
			actual[row * N + col] = sum;
		}
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time = 0;
	cudaEventElapsedTime(&time, start, end);

	double error = 0;
	for (int k = 0; k < ROWS * COLS; k++)
	{
		cout << k << ")"
			 << "Expected value = " << actual[k] << " Actual value = " << hostC[k] << "\n";
		error += double(abs(actual[k] - hostC[k]));
	}

	error = sqrt(error);
	cout << "Error = " << error << "\n";
	cout << "The time required for it is: " << time << " seconds" << endl;

	delete[] hostA;
	delete[] hostB;
	delete[] hostC;
}

__global__ void matrixMultiplication(int *A, int *B, int *C, int N)
{

	int ROW = blockIdx.y * blockDim.y + threadIdx.y;
	int COL = blockIdx.x * blockDim.x + threadIdx.x;

	int sum = 0;
	if (ROW < N && COL < N)
	{
		for (int i = 0; i < N; i++)
		{
			sum += A[ROW * N + i] * B[i * N + COL];
		}
		__syncthreads();
		C[ROW * N + COL] = sum;
	}
}

void mm(int *A, int *B, int *C, int N)
{
	dim3 threadsPerblock(N, N);
	dim3 blocksPerGrid(1, 1);

	if (N * N > 512)
	{
		threadsPerblock.x = 512;
		threadsPerblock.y = 512;
		blocksPerGrid.x = ceil(double(N) / double(threadsPerblock.x));
		blocksPerGrid.y = ceil(double(N) / double(threadsPerblock.y));
	}

	cout << "Calling mult:"
		 << "\n";

	matrixMultiplication<<<blocksPerGrid, threadsPerblock>>>(A, B, C, N);
}