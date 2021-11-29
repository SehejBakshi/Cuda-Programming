#include <bits/stdc++.h>
using namespace std;

__global__ void kernel(int *gpu_bb, int *gpu_nn, int *gpu_aa, int *gpu_cc)
{
    int gpu_i = threadIdx.x + blockIdx.x * blockDim.x;
    if (*gpu_bb == gpu_aa[gpu_i])
    {
        *gpu_cc = 1;
    }
}

int binarySearch(int arr[], int l, int r, int x)
{
    if (r >= l)
    {
        int mid = l + (r - l) / 2;

        if (arr[mid] == x)
            return mid;

        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);

        return binarySearch(arr, mid + 1, r, x);
    }
    return -1;
}

int main()
{
    int n = 8;
    int a[n] = {13, 27, 15, 14, 33, 2, 24, 40};
    int b;
    int c = 0;

    //Kernel Variables
    int *g_p, *g_n, *g_b, *g_a, *g_c;

    //CUDA GRID BLOCK SIZE AND NUMBER OF BLOCKS
    int block_size = 32;
    int n_blocks = n / block_size + (n % block_size == 0 ? 0 : 1);

    size_t size = n * sizeof(int);

    // Memory Allocation
    cudaMalloc((void **)&g_p, sizeof(int));
    cudaMalloc((void **)&g_b, sizeof(int));
    cudaMalloc((void **)&g_c, sizeof(int));
    cudaMalloc((void **)&g_a, size);

    cout << "Number of elements: " << n << endl;
    for (int i = 0; i < n; i++)
    {
        // a[i] = rand() % 1000;
        cout << a[i] << " ";
    }
    cout << endl;
    cout << "Enter number you want to find: ";
    cin >> b;

    int result = binarySearch(a, 0, n - 1, b);

    cudaMemcpy(g_b, &b, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_n, &n, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_c, &c, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_a, &a, size, cudaMemcpyHostToDevice);

    // call kernel
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);
    kernel<<<n_blocks, block_size>>>(g_b, g_n, g_a, g_c);
    cudaEventRecord(gpu_stop);

    // Retrieve result from device and store it in host array
    cudaMemcpy(&b, g_b, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&n, g_n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c, g_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&a, g_a, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(gpu_stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, gpu_start, gpu_stop);

    cudaFree(g_b);
    cudaFree(g_n);
    cudaFree(g_c);
    cudaFree(g_a);

    if (result == -1)
    {
        cout << "The number is not found" << endl;
    }
    else
    {
        cout << "The number is found" << endl;
    }

    cout << "Time taken with GPU::" << milliseconds << endl;
    return 0;
}