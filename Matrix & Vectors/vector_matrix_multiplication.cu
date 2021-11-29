#include <bits/stdc++.h>
using namespace std;

int NUMBER_OF_ELEMENTS = 4;
int SIZE = NUMBER_OF_ELEMENTS * sizeof(int);
int VECTOR_SIZE = 4;

__global__ void kernel_multiplication(int *A, int *B, int *C, int N, int M);

int main()
{
	cudaEvent_t start, end;

	//allocate memory for host vectors
	int *hostA = (int *)malloc(VECTOR_SIZE * sizeof(int));
	int *hostB = (int *)malloc(SIZE * VECTOR_SIZE);
	int *hostC = (int *)malloc(VECTOR_SIZE * sizeof(int));

	int *deviceA, *deviceB, *deviceC;

	srand(time(0));
	int i, j;

	//initialize host vector by random elements
	for (i = 0; i < VECTOR_SIZE; i++)
	{
		hostA[i] = rand() % 10;
	}

	//initialize matrix by random elements
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		for (j = 0; j < VECTOR_SIZE; j++)
		{
			hostB[i * VECTOR_SIZE + j] = rand() % 10;
		}
	}

	cout << "A is:" << endl;
	for (i = 0; i < VECTOR_SIZE; i++)
	{
		cout << hostA[i] << " ";
	}
	cout << endl;

	cout << "B is:" << endl;
	for (i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		for (j = 0; j < VECTOR_SIZE; j++)
		{
			cout << hostB[i * VECTOR_SIZE + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	//allocate memory for device vectors
	cudaMalloc(&deviceA, VECTOR_SIZE * sizeof(int));
	cudaMalloc(&deviceB, NUMBER_OF_ELEMENTS * VECTOR_SIZE * sizeof(int));
	cudaMalloc(&deviceC, VECTOR_SIZE * sizeof(int));

	//kernel function
	cudaMemcpy(deviceA, hostA, VECTOR_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, SIZE * VECTOR_SIZE, cudaMemcpyHostToDevice);
	kernel_multiplication<<<NUMBER_OF_ELEMENTS, 1>>>(deviceA, deviceB, deviceC, NUMBER_OF_ELEMENTS, VECTOR_SIZE);
	cudaDeviceSynchronize();
	cudaMemcpy(hostC, deviceC, VECTOR_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);

	double error = 0;

	int *answer = (int *)malloc(VECTOR_SIZE * sizeof(int));
	for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
	{
		int sum = 0;
		for (int j = 0; j < VECTOR_SIZE; j++)
		{
			sum += hostA[j] * hostB[i * VECTOR_SIZE + j];
		}
		answer[i] = sum;
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time = 0;
	cudaEventElapsedTime(&time, start, end);

	for (int k = 0; k < VECTOR_SIZE; k++)
	{
		cout << k << ")"
			 << "Expected value = " << answer[k] << " Actual value = " << hostC[k] << "\n";
		error += double(abs(answer[k] - hostC[k]));
	}

	error = sqrt(error);
	cout << "error = " << error << "\n";
	cout << "The time required for it is: " << time << " seconds" << endl;

	delete[] hostA;
	delete[] hostB;
	delete[] hostC;

	return cudaDeviceSynchronize();
}

__global__ void kernel_multiplication(int *A, int *B, int *C, int N, int M)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int sum = 0;

	if (index < N)
	{
		for (int i = 0; i < M; i++)
			sum += A[i] * B[(index * M) + i];
		C[index] = sum;
	}
}