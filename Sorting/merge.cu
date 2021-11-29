#include <bits/stdc++.h>
using namespace std;

__device__ void CudaMerge(int *values, int *results, int l, int r, int u)
{
    int i, j, k;
    i = l;
    j = r;
    k = l;
    while (i < r && j < u)
    {
        if (values[i] <= values[j])
        {
            results[k] = values[i];
            i++;
        }
        else
        {
            results[k] = values[j];
            j++;
        }
        k++;
    }

    while (i < r)
    {
        results[k] = values[i];
        i++;
        k++;
    }

    while (j < u)
    {
        results[k] = values[j];
        j++;
        k++;
    }
    for (k = l; k < u; k++)
    {
        values[k] = results[k];
    }
}

// Function to generate the sublists of the array to sort.
// It uses flag __global__ because the function is called from the main. It is also call Kernel and use a specific call.

__global__ static void CudaMergeSort(int *values, int *results, int dim)
{
    extern __shared__ int shared[];

    const unsigned int tid = threadIdx.x;
    int k, u, i;
    shared[tid] = values[tid];

    __syncthreads();
    k = 1;
    while (k <= dim)
    {
        i = 0;
        while (i + k < dim)
        {
            u = i + k * 2;
            ;
            if (u > dim)
            {
                u = dim + 1;
            }
            CudaMerge(shared, results, i, i + k, u);
            i = i + k * 2;
        }
        k = k * 2;

        __syncthreads();
    }

    values[tid] = shared[tid];
}

void mergeSort(int arr[], int p, int q);

int main()
{
    int N = 100;
    float elapsed;

    // int a[N] = {13, 27, 15, 14, 33, 2, 24, 40};
    int a[N];

    int *dvalues, *results;

    cout << "Elements for N = 100:\n";
    for (int i = 0; i < N; i++)
    {
        a[i] = rand() % 1000;
        cout << a[i] << " ";
    }
    cout << endl;

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    cudaMalloc((void **)&dvalues, sizeof(int) * N);
    cudaMemcpy(dvalues, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&results, sizeof(int) * N);
    cudaMemcpy(results, a, sizeof(int) * N, cudaMemcpyHostToDevice);

    CudaMergeSort<<<1, N, sizeof(int) * N * 2>>>(dvalues, results, N);

    cudaFree(dvalues);
    cudaMemcpy(a, results, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaFree(results);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << endl;

    cout << "Sorted elements are: ";
    for (int i = 0; i < N; i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;

    cout << "The time required for it is: " << elapsed << " seconds" << endl;

    cudaDeviceReset();
    cudaThreadExit();

    return 0;
}

void merge(int arr[], int p, int q, int r)
{

    int i, j, k;
    int n1 = q - p + 1;
    int n2 = r - q;

    int L[n1], M[n2];

    for (int i = 0; i < n1; i++)
    {
        L[i] = arr[p + i];
    }

    for (int j = 0; j < n2; j++)
    {
        M[j] = arr[q + 1 + j];
    }

    i = 0;
    j = 0;
    k = p;

    while (i < n1 && j < n2)
    {
        if (L[i] <= M[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = M[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = M[j];
        j++;
        k++;
    }
}

void mergeSort(int arr[], int p, int q)
{

    if (p < q)
    {
        int mitad = (p + q) / 2;

        mergeSort(arr, p, mitad);
        mergeSort(arr, mitad + 1, q);
        merge(arr, p, mitad, q);
    }
}
