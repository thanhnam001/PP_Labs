// Last update: 16/12/2020
#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__device__ int bCount = 0;
volatile __device__ int bCount1 = 0;
__global__ void scanKernel(uint32_t* in, int n, uint32_t* out, volatile uint32_t* bSums){
    // Local scan
    extern __shared__ uint32_t s_data[];
    __shared__ int bi;
    if(threadIdx.x == 0)
        bi = atomicAdd(&bCount, 1);
    __syncthreads();
    int i1 = bi * 2 * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x;
    s_data[threadIdx.x] = (0 < i1 && i1 < n) ? in[i1 - 1] : 0;
    s_data[threadIdx.x + blockDim.x] = (i2 < n) ? in[i2 - 1] :0;
    __syncthreads();

    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1;
        if ( s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride;
        if (s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }
    if (bSums != NULL && threadIdx.x == 0)
        bSums[bi] = s_data[2 * blockDim.x - 1];
    __syncthreads();

    if(threadIdx.x == 0){
        if(bi > 0){
            while(bCount1 < bi){}
            bSums[bi] += bSums[bi - 1];
            __threadfence();
        }
        bCount1 += 1;
    }
    __syncthreads();
    // Calculate block sum
    if (i1 < n)
        out[i1] = s_data[threadIdx.x] + ((bi > 0) ? bSums[bi - 1] : 0);
    if (i2 < n)
        out[i2] = s_data[threadIdx.x + blockDim.x] + ((bi > 0) ? bSums[bi - 1] : 0);
}

__global__ void rankAndSwap(uint32_t *src, uint32_t *dst, uint32_t *bits, uint32_t *nOnesBefore, int nZeros, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        int rank = (bits[i] == 0) ? i - nOnesBefore[i] : nZeros + nOnesBefore[i];
        dst[rank] = src[i];
    }
}

__global__ void extractBits(uint32_t *d_in, uint32_t *d_out, int bitIdx, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        d_out[i] = (d_in[i] >> bitIdx) & 1;
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    uint32_t *d_src, *d_dst; // For ranking and swapping 
    uint32_t *d_bits, *d_nOnesBefore; // d_bits for bits, d_nOnesBefore for exclusive scanned bits
    uint32_t *d_blkSums;
    size_t nBytes = n * sizeof(uint32_t);
    const int z = 0;
    uint32_t totalOnes;
    uint32_t lastBit;
    CHECK(cudaMalloc(&d_bits, nBytes));
    CHECK(cudaMalloc(&d_nOnesBefore, nBytes));

    CHECK(cudaMalloc(&d_src, nBytes));
    CHECK(cudaMalloc(&d_dst, nBytes));
    CHECK(cudaMemcpy(d_src, in, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dst, out, nBytes, cudaMemcpyHostToDevice));
    
    dim3 gridSize((n - 1)/blockSize + 1);
    dim3 gridSize1((n - 1)/blockSize/2 + 1);
    if (gridSize.x > 1){
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    }
    else
        d_blkSums = NULL;

    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx ++){
        CHECK(cudaMemcpyToSymbol(bCount , &z, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(bCount1, &z, sizeof(int)));

        extractBits<<<gridSize,blockSize>>>(d_src, d_bits, bitIdx, n);

        size_t smem = 2 * blockSize * sizeof(uint32_t);
        
        scanKernel<<<gridSize1,blockSize,smem>>>(d_bits, n, d_nOnesBefore, d_blkSums);
        
        CHECK(cudaMemcpy(&lastBit, d_bits + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&totalOnes, d_nOnesBefore + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        int nZeros = n - totalOnes - lastBit;
        rankAndSwap<<<gridSize,blockSize>>>(d_src, d_dst, d_bits, d_nOnesBefore, nZeros, n);
        CHECK(cudaGetLastError());
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }
    CHECK(cudaMemcpy(out, d_src, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_nOnesBefore));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_blkSums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    // int n = 500; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        // in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    // printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    // printArray(correctOut, 1000); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    // printArray(out, 1000); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
