// nvcc -o a.out <program_name> - How to run
#include <bits/stdc++.h>
using namespace std;

__global__ void findMax(float *input)
{
    int tid = threadIdx.x;
    auto step = 1;
    int number_of_threads = blockDim.x;

    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            int first = tid * step * 2;
            int second = first + step;
            if (input[second] > input[first])
                input[first] = input[second];
        }
        step *= 2;
        number_of_threads /= 2;
    }
}

__global__ void findMin(float *input)
{
    int thread_id = threadIdx.x;
    auto step = 1;
    int number_of_threads = blockDim.x;
    while (number_of_threads > 0)
    {
        if (thread_id < number_of_threads)
        {
            int first = thread_id * step * 2;
            int second = first + step;
            if (input[second] < input[first])
                input[first] = input[second];
        }
        step = step * 2;
        number_of_threads /= 2;
    }
}

__global__ void findAvg(float *input)
{
    const int tid = threadIdx.x;
    auto step = 1;
    int number_of_threads = blockDim.x;
    int totalElements = number_of_threads * 2;
    while (number_of_threads > 0)
    {
        if (tid < number_of_threads)
        {
            const int first = tid * step * 2;
            const int second = first + step;
            input[first] = input[first] + input[second];
        }
        step = step * 2;
        number_of_threads = number_of_threads / 2;
    }
    input[0] = input[0] / totalElements;
}

int main()
{
    cudaEvent_t start, end;
    // int n = 8;
    int n;
    cin >> n;
    // float arr[] = {13, 27, 15, 14, 33, 2, 24, 40};
    float *arr = new float[n];
    int size = n * sizeof(float);
    cout << "Number of elements: " << n << endl;
    for (int i = 0; i < n; i++)
    {
        arr[i] = rand() % 1000;
        cout << arr[i] << " ";
    }
    cout << endl;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    //Calculating maximum number in array arr
    float *arr_max, result_max;
    cudaMalloc(&arr_max, size);
    cudaMemcpy(arr_max, arr, size, cudaMemcpyHostToDevice);
    //Launch max kernel on GPU with n/2 threads
    findMax<<<1, n / 2>>>(arr_max);
    cudaMemcpy(&result_max, arr_max, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "\nThe maximum element is " << result_max << endl;

    //Calculating minimum number in array arr
    float *arr_min, result_min;
    cudaMalloc(&arr_min, size);
    cudaMemcpy(arr_min, arr, size, cudaMemcpyHostToDevice);
    //Launch min kernel on GPU with n/2 threads
    findMin<<<1, n / 2>>>(arr_min);
    cudaMemcpy(&result_min, arr_min, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "The minimum element is " << result_min << endl;

    //Calculating average of numbers in array arr
    float *arr_avg, result_avg;
    cudaMalloc(&arr_avg, size);
    cudaMemcpy(arr_avg, arr, size, cudaMemcpyHostToDevice);
    //Launch avg kernel on GPU with n/2 threads
    findAvg<<<1, n / 2>>>(arr_avg);
    cudaMemcpy(&result_avg, arr_avg, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "The average of elements is " << result_avg << endl;

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    cout << "The time required for it is: " << time << " seconds" << endl;

    cudaFree(arr_max);
    cudaFree(arr_min);
    cudaFree(arr_avg);
    return 0;
}
