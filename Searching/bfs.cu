#include <bits/stdc++.h>
#define NUM_NODES 5

using namespace std;

typedef struct
{
	int start;	// Index of first adjacent node in Ea
	int length; // Number of adjacent nodes
} Node;

__global__ void CUDA_BFS_KERNEL(Node *Va, int *Ea, bool *Fa, bool *Xa, int *Ca, bool *done)
{

	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id > NUM_NODES)
		*done = false;

	if (Fa[id] == true && Xa[id] == false)
	{
		printf("%d ", id); //This printf gives the order of vertices in BFS
		Fa[id] = false;
		Xa[id] = true;
		__syncthreads();

		int start = Va[id].start;
		int end = start + Va[id].length;
		for (int i = start; i < end; i++)
		{
			int nid = Ea[i];

			if (Xa[nid] == false)
			{
				Ca[nid] = Ca[id] + 1;
				Fa[nid] = true;
				*done = false;
			}
		}
	}
}

// The BFS frontier corresponds to all the nodes being processed at the current level.

int main()
{
	cudaEvent_t start, end;

	Node node[NUM_NODES];

	int edges[NUM_NODES];

	node[0].start = 0;
	node[0].length = 2;

	node[1].start = 2;
	node[1].length = 1;

	node[2].start = 3;
	node[2].length = 1;

	node[3].start = 4;
	node[3].length = 1;

	node[4].start = 5;
	node[4].length = 0;

	edges[0] = 1;
	edges[1] = 2;
	edges[2] = 4;
	edges[3] = 3;
	edges[4] = 4;

	bool frontier[NUM_NODES] = {false};
	bool visited[NUM_NODES] = {false};
	int cost[NUM_NODES] = {0};

	int source = 0;
	frontier[source] = true;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	Node *Va;
	cudaMalloc((void **)&Va, sizeof(Node) * NUM_NODES);
	cudaMemcpy(Va, node, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

	int *Ea;
	cudaMalloc((void **)&Ea, sizeof(Node) * NUM_NODES);
	cudaMemcpy(Ea, edges, sizeof(Node) * NUM_NODES, cudaMemcpyHostToDevice);

	bool *Fa;
	cudaMalloc((void **)&Fa, sizeof(bool) * NUM_NODES);
	cudaMemcpy(Fa, frontier, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

	bool *Xa;
	cudaMalloc((void **)&Xa, sizeof(bool) * NUM_NODES);
	cudaMemcpy(Xa, visited, sizeof(bool) * NUM_NODES, cudaMemcpyHostToDevice);

	int *Ca;
	cudaMalloc((void **)&Ca, sizeof(int) * NUM_NODES);
	cudaMemcpy(Ca, cost, sizeof(int) * NUM_NODES, cudaMemcpyHostToDevice);

	int num_blks = 1;
	int threads = 5;

	bool done;
	bool *d_done;
	cudaMalloc((void **)&d_done, sizeof(bool));

	int count = 0;

	cout << "Order: ";
	do
	{
		count++;
		done = true;
		cudaMemcpy(d_done, &done, sizeof(bool), cudaMemcpyHostToDevice);
		CUDA_BFS_KERNEL<<<num_blks, threads>>>(Va, Ea, Fa, Xa, Ca, d_done);
		cudaMemcpy(&done, d_done, sizeof(bool), cudaMemcpyDeviceToHost);

	} while (!done);

	cudaMemcpy(cost, Ca, sizeof(int) * NUM_NODES, cudaMemcpyDeviceToHost);
	cout << endl;

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float time = 0;
	cudaEventElapsedTime(&time, start, end);

	cout << "Number of times the kernel is called: " << count << endl;

	cout << "Cost: ";
	for (int i = 0; i < NUM_NODES; i++)
		cout << cost[i] << " ";
	cout << endl;

	cout << "The time required for it is: " << time << " seconds" << endl;
}