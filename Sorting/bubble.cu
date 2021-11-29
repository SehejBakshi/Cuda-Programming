#include <bits/stdc++.h>
using namespace std;

template <typename T>
struct ShouldSwap
{
    __host__ __device__ virtual bool operator()(const T left, const T right) const;
};

template <typename T>
__host__ __device__ __inline__ void swap(T *a, T *r);

template <typename T>
__global__ void bubbleSort(T *v, const unsigned int n, ShouldSwap<T> shouldSwap);

int main(int argc, char **argv)
{
    cudaEvent_t start, end;
    const unsigned int size = 100;

    // int h_v[size] = {13, 27, 15, 14, 33, 2, 24, 40};
    int h_v[size];
    cout << "Number of elements: " << size << endl;
    for (int i = 0; i < size; i++)
    {
        h_v[i] = rand() % 1000;
        cout << h_v[i] << " ";
    }
    cout << endl;

    int *d_v = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);

    cudaMalloc((void **)&d_v, size * sizeof(int));
    cudaMemcpy(d_v, h_v, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grdDim(1, 1, 1);
    dim3 blkDim(size / 2, 1, 1);

    ShouldSwap<int> shouldSwap;

    bubbleSort<int><<<grdDim, blkDim>>>(d_v, size, shouldSwap);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(h_v, d_v, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_v);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float time = 0;
    cudaEventElapsedTime(&time, start, end);

    cout << endl;

    cout << "Sorted array is: ";
    for (int i = 0; i < size; i++)
    {
        std::cout << h_v[i] << " ";
    }
    std::cout << std::endl;

    cout << "The time required for it is: " << time << " seconds" << endl;
    return 0;
}

template <typename T>
__host__ __device__ bool ShouldSwap<T>::operator()(const T left, const T right) const
{
    return left > right;
}

template <typename T>
__host__ __device__ __inline__ void swap(T *a, T *b)
{
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

template <typename T>
__global__ void bubbleSort(T *v, const unsigned int n, ShouldSwap<T> shouldSwap)
{
    const unsigned int tIdx = threadIdx.x;

    for (unsigned int i = 0; i < n; i++)
    {

        unsigned int offset = i % 2;
        unsigned int indiceGauche = 2 * tIdx + offset;
        unsigned int indiceDroite = indiceGauche + 1;

        if (indiceDroite < n)
        {
            if (shouldSwap(v[indiceGauche], v[indiceDroite]))
            {
                swap<T>(&v[indiceGauche], &v[indiceDroite]);
            }
        }
        __syncthreads();
    }
}