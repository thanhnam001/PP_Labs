#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 9
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
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

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

__global__ void blurImgKernel1(uchar3 * inPixels, int width, int height, 
        float * filter, int filterWidth, 
        uchar3 * outPixels)
{
	// TODO
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int halfFilterWidth = (filterWidth - 1)/2;
	if(col < width && row < height){
		float3 outPixel = make_float3(0, 0, 0);
		for (int filterR = 0; filterR < filterWidth; filterR++){
			for (int filterC = 0; filterC < filterWidth; filterC++){
				float filterVal = filter[filterR*filterWidth + filterC];
				int inPixelsR = row - halfFilterWidth + filterR;
				int inPixelsC = col - halfFilterWidth + filterC;
				inPixelsR = min(max(0, inPixelsR), height - 1);
				inPixelsC = min(max(0, inPixelsC), width - 1);
				uchar3 inPixel = inPixels[inPixelsR*width + inPixelsC];
				outPixel.x += filterVal * inPixel.x;
				outPixel.y += filterVal * inPixel.y;
				outPixel.z += filterVal * inPixel.z;
			}
		}
		outPixels[row*width + col] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
	}
}

__global__ void blurImgKernel2(uchar3 * inPixels, int width, int height, 
        float * filter, int filterWidth, 
        uchar3 * outPixels)
{
	// TODO
	extern __shared__ uchar3 s_inPixels[];
	int halfFilterWidth = filterWidth >> 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < width && row < height){
		int i = row * width + col;
		// center
		int center = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + halfFilterWidth + threadIdx.x;
		s_inPixels[center] = inPixels[i];

		int topRow = max(row - halfFilterWidth, 0);
		int bottomRow = min(row + blockDim.y, height - 1);
		int leftCol = max(col - halfFilterWidth, 0);
		int rightCol = min(col + blockDim.x, width - 1);
		if(threadIdx.y < halfFilterWidth) {
			// top
			int top = (blockDim.x + filterWidth - 1) * threadIdx.y + halfFilterWidth + threadIdx.x;
			s_inPixels[top] = inPixels[topRow * width + col];
			// bottom
			int bottom = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + halfFilterWidth + threadIdx.x;
			s_inPixels[bottom] = inPixels[bottomRow * width + col];
		}
		if(threadIdx.x < halfFilterWidth){
			//left
			int left = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + threadIdx.x;
			s_inPixels[left] = inPixels[row * width + leftCol];
			//right
			int right = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[right] = inPixels[row * width + rightCol];
		}
		// 4 corner
		if(threadIdx.x < halfFilterWidth && threadIdx.y < halfFilterWidth) {
			// top left corner
			int topLeftCorner = (blockDim.x + filterWidth - 1) * threadIdx.y + threadIdx.x;
			s_inPixels[topLeftCorner] = inPixels[topRow * width + leftCol];
			// top right corner
			int topRightCorner = (blockDim.x + filterWidth - 1) * threadIdx.y + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[topRightCorner] = inPixels[topRow * width + rightCol];
			// bottom left corner
			int bottomLeftCorner = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + threadIdx.x;
			s_inPixels[bottomLeftCorner] = inPixels[bottomRow * width + leftCol];
			// bottom right corner
			int bottomRightCorner = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[bottomRightCorner] = inPixels[bottomRow * width + rightCol];
		}
		__syncthreads();
		float3 outPixel = make_float3(0, 0, 0);
		for (int filterR = 0; filterR < filterWidth; filterR++){
			for (int filterC = 0; filterC < filterWidth; filterC++){
				float filterVal = filter[filterR * filterWidth + filterC];
				int s_inPixelsR = threadIdx.y + filterR;
				int s_inPixelsC = threadIdx.x + filterC;
				uchar3 s_inPixel = s_inPixels[s_inPixelsR * (blockDim.x + filterWidth - 1) + s_inPixelsC];
				outPixel.x += filterVal * s_inPixel.x;
				outPixel.y += filterVal * s_inPixel.y;
				outPixel.z += filterVal * s_inPixel.z;
			}
		}
		outPixels[i] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
	}
}

__global__ void blurImgKernel3(uchar3 * inPixels, int width, int height, 
        int filterWidth, 
        uchar3 * outPixels)
{
	// TODO
	extern __shared__ uchar3 s_inPixels[];
	int halfFilterWidth = filterWidth >> 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col < width && row < height){
		int i = row * width + col;
		// center
		int center = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + halfFilterWidth + threadIdx.x;
		s_inPixels[center] = inPixels[i];

		int topRow = max(row - halfFilterWidth, 0);
		int bottomRow = min(row + blockDim.y, height - 1);
		int leftCol = max(col - halfFilterWidth, 0);
		int rightCol = min(col + blockDim.x, width - 1);
		if(threadIdx.y < halfFilterWidth) {
			// top
			int top = (blockDim.x + filterWidth - 1) * threadIdx.y + halfFilterWidth + threadIdx.x;
			s_inPixels[top] = inPixels[topRow * width + col];
			// bottom
			int bottom = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + halfFilterWidth + threadIdx.x;
			s_inPixels[bottom] = inPixels[bottomRow * width + col];
		}
		if(threadIdx.x < halfFilterWidth){
			//left
			int left = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + threadIdx.x;
			s_inPixels[left] = inPixels[row * width + leftCol];
			//right
			int right = (blockDim.x + filterWidth - 1) * (threadIdx.y + halfFilterWidth) + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[right] = inPixels[row * width + rightCol];
		}
		// 4 corner
		if(threadIdx.x < halfFilterWidth && threadIdx.y < halfFilterWidth) {
			// top left corner
			int topLeftCorner = (blockDim.x + filterWidth - 1) * threadIdx.y + threadIdx.x;
			s_inPixels[topLeftCorner] = inPixels[topRow * width + leftCol];
			// top right corner
			int topRightCorner = (blockDim.x + filterWidth - 1) * threadIdx.y + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[topRightCorner] = inPixels[topRow * width + rightCol];
			// bottom left corner
			int bottomLeftCorner = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + threadIdx.x;
			s_inPixels[bottomLeftCorner] = inPixels[bottomRow * width + leftCol];
			// bottom right corner
			int bottomRightCorner = (blockDim.x + filterWidth - 1) * (threadIdx.y + blockDim.y + halfFilterWidth) + halfFilterWidth + threadIdx.x + blockDim.x;
			s_inPixels[bottomRightCorner] = inPixels[bottomRow * width + rightCol];
		}
		__syncthreads();
		float3 outPixel = make_float3(0, 0, 0);
		for (int filterR = 0; filterR < filterWidth; filterR++){
			for (int filterC = 0; filterC < filterWidth; filterC++){
				float filterVal = dc_filter[filterR * filterWidth + filterC];
				int s_inPixelsR = threadIdx.y + filterR;
				int s_inPixelsC = threadIdx.x + filterC;
				uchar3 s_inPixel = s_inPixels[s_inPixelsR * (blockDim.x + filterWidth - 1) + s_inPixelsC];
				outPixel.x += filterVal * s_inPixel.x;
				outPixel.y += filterVal * s_inPixel.y;
				outPixel.z += filterVal * s_inPixel.z;
			}
		}
		outPixels[i] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
	}
}							

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
        uchar3 * outPixels,
        bool useDevice=false, dim3 blockSize=dim3(1, 1), int kernelType=1)
{
	if (useDevice == false)
	{
		for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
		{
			for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
			{
				float3 outPixel = make_float3(0, 0, 0);
				for (int filterR = 0; filterR < filterWidth; filterR++)
				{
					for (int filterC = 0; filterC < filterWidth; filterC++)
					{
						float filterVal = filter[filterR*filterWidth + filterC];
						int inPixelsR = outPixelsR - filterWidth/2 + filterR;
						int inPixelsC = outPixelsC - filterWidth/2 + filterC;
						inPixelsR = min(max(0, inPixelsR), height - 1);
						inPixelsC = min(max(0, inPixelsC), width - 1);
						uchar3 inPixel = inPixels[inPixelsR*width + inPixelsC];
						outPixel.x += filterVal * inPixel.x;
						outPixel.y += filterVal * inPixel.y;
						outPixel.z += filterVal * inPixel.z;
					}
				}
				outPixels[outPixelsR*width + outPixelsC] = make_uchar3(outPixel.x, outPixel.y, outPixel.z); 
			}
		}
	}
	else // Use device
	{
		GpuTimer timer;
		
		printf("\nKernel %i, ", kernelType);
		// Allocate device memories
		uchar3 * d_inPixels, * d_outPixels;
		float * d_filter;
		size_t pixelsSize = width * height * sizeof(uchar3);
		size_t filterSize = filterWidth * filterWidth * sizeof(float);
		CHECK(cudaMalloc(&d_inPixels, pixelsSize));
		CHECK(cudaMalloc(&d_outPixels, pixelsSize));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaMalloc(&d_filter, filterSize));
		}

		// Copy data to device memories
		CHECK(cudaMemcpy(d_inPixels, inPixels, pixelsSize, cudaMemcpyHostToDevice));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));
		}
		else
		{
			// TODO: copy data from "filter" (on host) to "dc_filter" (on CMEM of device)
			CHECK(cudaMemcpyToSymbol(dc_filter, filter, filterSize));
		}

		// Call kernel
		dim3 gridSize((width-1)/blockSize.x + 1, (height-1)/blockSize.y + 1);
		size_t s = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uchar3);
		printf("block size %ix%i, grid size %ix%i\n", blockSize.x, blockSize.y, gridSize.x, gridSize.y);
		timer.Start();
		if (kernelType == 1)
		{
			// TODO: call blurImgKernel1
			blurImgKernel1<<<gridSize, blockSize>>>(d_inPixels,width,height,d_filter,filterWidth,d_outPixels);
		}
		else if (kernelType == 2)
		{
			// TODO: call blurImgKernel2
			blurImgKernel2<<<gridSize, blockSize, s>>>(d_inPixels,width,height,d_filter,filterWidth,d_outPixels);
		}
		else
		{
			// TODO: call blurImgKernel3
			blurImgKernel3<<<gridSize, blockSize, s>>>(d_inPixels,width,height,filterWidth,d_outPixels);
		}
		timer.Stop();
		float time = timer.Elapsed();
		printf("Kernel time: %f ms\n", time);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memory
		CHECK(cudaMemcpy(outPixels, d_outPixels, pixelsSize, cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_inPixels));
		CHECK(cudaFree(d_outPixels));
		if (kernelType == 1 || kernelType == 2)
		{
			CHECK(cudaFree(d_filter));
		}
	}
	
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

void printError(uchar3 * deviceResult, uchar3 * hostResult, int width, int height)
{
	float err = computeError(deviceResult, hostResult, width * height);
	printf("Error: %f\n", err);
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
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
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("CMEM: %lu bytes\n", devProv.totalConstMem);
    printf("L2 cache: %i bytes\n", devProv.l2CacheSize);
    printf("SMEM / one SM: %lu bytes\n", devProv.sharedMemPerMultiprocessor);

    printf("****************************\n");

}

int main(int argc, char ** argv)
{
	if (argc !=3 && argc != 5)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	printDeviceInfo();

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("\nImage size (width x height): %i x %i\n", width, height);

	// Set up a simple filter with blurring effect 
	int filterWidth = FILTER_WIDTH;
	float * filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// Blur input image not using device
	uchar3 * correctOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	blurImg(inPixels, width, height, filter, filterWidth, correctOutPixels);
	
    // Blur input image using device, kernel 1
    dim3 blockSize(32, 32); // Default
	if (argc == 5)
	{
		blockSize.x = atoi(argv[3]);
		blockSize.y = atoi(argv[4]);
	}	
	uchar3 * outPixels1 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels1, true, blockSize, 1);
	printError(outPixels1, correctOutPixels, width, height);
	
	// Blur input image using device, kernel 2
	uchar3 * outPixels2 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels2, true, blockSize, 2);
	printError(outPixels2, correctOutPixels, width, height);

	// Blur input image using device, kernel 3
	uchar3 * outPixels3 = (uchar3 *)malloc(width * height * sizeof(uchar3));
	blurImg(inPixels, width, height, filter, filterWidth, outPixels3, true, blockSize, 3);
	printError(outPixels3, correctOutPixels, width, height);

    // Write results to files
    char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(correctOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(outPixels1, width, height, concatStr(outFileNameBase, "_device1.pnm"));
	writePnm(outPixels2, width, height, concatStr(outFileNameBase, "_device2.pnm"));
	writePnm(outPixels3, width, height, concatStr(outFileNameBase, "_device3.pnm"));

	// Free memories
	free(inPixels);
	free(filter);
	free(correctOutPixels);
	free(outPixels1);
	free(outPixels2);
	free(outPixels3);
}
