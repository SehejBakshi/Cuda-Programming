#include <bits/stdc++.h>
using namespace std;

int NUMBER_OF_ELEMENTS = 1 << 5;
int SIZE = NUMBER_OF_ELEMENTS * sizeof(int);

__global__ void kernel_sum(int *A, int *B, int *C, int NUMBERofELEMENTS);

void sum(int *A, int *B, int *C, int n_el);

int main()
{
	cudaEvent_t start, end;

	//allocate memory for host vectors
	int *hostA = (int *)malloc(SIZE);
	int *hostB = (int *)malloc(SIZE);
	int *hostC = (int *)malloc(SIZE);

	int *deviceA, *deviceB, *deviceC;

	srand(time(0));
	int i;
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		hostA[i] = rand() % 10;
		hostB[i] = rand() % 10;
	}

	cout << "Vector A:" << endl;
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		cout << hostA[i] << " ";
	}
	cout << endl;

	cout << "Vector B:" << endl;
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		cout << hostB[i] << " ";
	}
	cout << endl;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	//allocate memory for device vectors
	cudaMalloc(&deviceA, SIZE);
	cudaMalloc(&deviceB, SIZE);
	cudaMalloc(&deviceC, SIZE);

	//kernel function
	cudaMemcpy(deviceA, hostA, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, SIZE, cudaMemcpyHostToDevice);

	sum(deviceA, deviceB, deviceC, NUMBER_OF_ELEMENTS);

	cudaMemcpy(hostC, deviceC, SIZE, cudaMemcpyDeviceToHost);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time = 0;
	cudaEventElapsedTime(&time, start, end);

	double error = 0;
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		double diff = double((hostA[i] + hostB[i]) - hostC[i]);
		error += diff;

		cout << "A+B = " << hostA[i] + hostB[i] << ", ";
		cout << "C = " << hostC[i] << "\n";
	}

	error = sqrt(error);
	cout << "error = " << error << endl;
	cout << "The time required for it is: " << time << " seconds" << endl;

	delete[] hostA;
	delete[] hostB;
	delete[] hostC;

	return cudaDeviceSynchronize();
}

void sum(int *A, int *B, int *C, int n_el)
{
	int threadsPerblock, blocksperGrid;

	if (n_el < 512)
	{
		threadsPerblock = n_el;
		blocksperGrid = 1;
	}
	else
	{
		threadsPerblock = 512;
		blocksperGrid = ceil(double(n_el) / double(threadsPerblock));
	}

	kernel_sum<<<blocksperGrid, threadsPerblock>>>(A, B, C, n_el);
}

__global__ void kernel_sum(int *A, int *B, int *C, int NUMBERofELEMENTS)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < NUMBERofELEMENTS)
		C[index] = A[index] + B[index];
}