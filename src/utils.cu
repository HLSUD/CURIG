/*
  Utility functions
  1. prefix_scan
  2. get_max_min
  3. rescale
  4. shift_and_scale
  5. matrix transpose
  6. GPU info
  7. GPU memory status
*/

#include "utils.h"
#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "dataType.h"

void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag)
{
  /*
    n - number of elements
    flag - 1 inclusive, 0 exclusive
    thrust::inclusive_scan(d_arr, d_arr + n, d_res);
  */
  thrust::device_ptr<PCS> d_ptr(d_arr); // not convert
  thrust::device_ptr<PCS> d_result(d_res);

  if (flag)
    thrust::inclusive_scan(d_ptr, d_ptr + n, d_result); 
  else
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);
}

void get_max_min(PCS &max, PCS &min, PCS *d_array, int n)
{
  /*
    Get the maximum and minimum of array by thrust
    d_array - array on device
    n - length of array
  */
  thrust::device_ptr<PCS> d_ptr = thrust::device_pointer_cast(d_array);
  max = *(thrust::max_element(d_ptr, d_ptr + n));

  min = *(thrust::min_element(d_ptr, d_ptr + n));
}

// real and complex array scaling
__global__ void rescaling_real(PCS *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx] *= scale_ratio;
  }
}

__global__ void rescaling_complex(CUCPX *x, PCS scale_ratio, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    x[idx].x *= scale_ratio;
    x[idx].y *= scale_ratio;
  }
}

void rescaling_real_invoker(PCS *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_real<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

void rescaling_complex_invoker(CUCPX *d_x, PCS scale_ratio, int N)
{
  int blocksize = 512;
  rescaling_complex<<<(N - 1) / blocksize + 1, blocksize>>>(d_x, scale_ratio, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void shift_and_scale(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  int idx;
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < M; idx += gridDim.x * blockDim.x)
  {
    d_u[idx] = (d_u[idx] - i_center) / gamma;
  }
  for (idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += gridDim.x * blockDim.x)
  {
    d_x[idx] = (d_x[idx] - o_center) * gamma;
  }
}

void shift_and_scale_invoker(PCS i_center, PCS o_center, PCS gamma, PCS *d_u, PCS *d_x, int M, int N)
{
  // Specified for nu to nu fourier transform
  int blocksize = 512;
  shift_and_scale<<<(max(M, N) - 1) / blocksize + 1, blocksize>>>(i_center, o_center, gamma, d_u, d_x, M, N);
  CHECK(cudaDeviceSynchronize());
}

__global__ void transpose(PCS *odata, PCS *idata, int width, int height)
{
  //* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
  // refer https://github.com/JonathanWatkins/CUDA/blob/master/NvidiaCourse/Exercises/transpose/transpose.cu
  __shared__ PCS block[BLOCKSIZE][BLOCKSIZE];

  // read the matrix tile into shared memory
  // load one element per thread from device memory (idata) and store it
  // in transposed order in block[][]
  unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  if ((xIndex < width) && (yIndex < height))
  {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  // synchronise to ensure all writes to block[][] have completed
  __syncthreads();

  // write the transposed matrix tile to global memory (odata) in linear order
  xIndex = blockIdx.y * blockDim.x + threadIdx.x;
  yIndex = blockIdx.x * blockDim.y + threadIdx.y;
  if ((xIndex < height) && (yIndex < width))
  {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
  // __syncthreads();
}


int matrix_transpose_invoker(PCS *d_arr, int width, int height)
{
  int ier = 0;
  int blocksize = BLOCKSIZE;
  dim3 block(blocksize, blocksize);
  dim3 grid((width - 1) / blocksize + 1, (height - 1) / blocksize + 1);
  PCS *temp_o;
  checkCudaErrors(cudaMalloc((void**)&temp_o,sizeof(PCS)*width*height));
  transpose<<<grid, block>>>(temp_o, d_arr, width, height);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(d_arr,temp_o,sizeof(PCS)*width*height,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaFree(temp_o));
  return ier;
}

__global__ void matrix_elementwise_multiply(CUCPX *a, PCS *b, int N){
  int idx;
  for(idx = threadIdx.x+blockIdx.x*blockDim.x; idx<N; idx+=gridDim.x*blockDim.x){
    a[idx].x = a[idx].x*b[idx];
    a[idx].y = a[idx].y*b[idx];
  }
}

int matrix_elementwise_multiply_invoker(CUCPX *a, PCS *b, int N){
  int ier=0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N-1)/blocksize+1,blocksize>>>(a,b,N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}

__global__ void matrix_elementwise_divide(CUCPX *a, PCS *b, int N){
  int idx;
  for(idx = threadIdx.x+blockIdx.x*blockDim.x; idx<N; idx+=gridDim.x*blockDim.x){
    a[idx].x = a[idx].x/b[idx];
    a[idx].y = a[idx].y/b[idx];
  }
}

int matrix_elementwise_divide_invoker(CUCPX *a, PCS *b, int N){
  int ier=0;
  int blocksize = 512;
  matrix_elementwise_multiply<<<(N-1)/blocksize+1,blocksize>>>(a,b,N);
  checkCudaErrors(cudaDeviceSynchronize());
  return ier;
}

void GPU_info()
{

  printf("Starting... \n");
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }

  if (deviceCount == 0)
    printf("There is no available device that support CUDA\n");
  else
    printf("Detected %d CUDA capable device(s)\n", deviceCount);

  int dev, driverVersion = 0, runtimeVersion = 0;

  dev = 0;

  printf("Input the device index:");
  std::cin >> dev;
  cudaSetDevice(dev);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device %d: %s\n", dev, deviceProp.name);

  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("CUDA Driver Version / Runtime Version       %d.%d / %d.%d\n",
         driverVersion / 1000, driverVersion % 1000, runtimeVersion / 1000, runtimeVersion % 1000);
  printf("CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf("Total amount of global memory:              %.2f MBytes\n", (float)deviceProp.totalGlobalMem / (pow(1024.0, 2)));
  printf("GPU clock rate:                             %.0f MHz\n", deviceProp.clockRate * 1e-3f);
  printf("Memory clock rate:                          %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
  printf("Memory Bus Width:                           %d-bit\n", deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize)
  {
    printf("L2 Cache Size:                          %d bytes\n", deviceProp.l2CacheSize);
  }
  printf("Total amount of constant memory:            %lu bytes\n", deviceProp.totalConstMem);
  printf("Total amount of shared memory per block:    %lu bytes\n", deviceProp.sharedMemPerBlock);
  printf("Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf("Warp size:                                  %d\n", deviceProp.warpSize);
  printf("Number of multiprocessors:                  %d\n", deviceProp.multiProcessorCount);
  printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  //printf("Maximum number of blocks per multiprocessor: %d\n",deviceProp.maxBlocksPerMultiProcessor);
  printf("Maximum number of thread per block:          %d\n", deviceProp.maxThreadsPerBlock);
  printf("Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf("Maximum sizes of each dimension of a grid:  %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
  printf("Maximum memory pitch:                       %lu bytes\n", deviceProp.memPitch);
}

int next235beven(int n, int b)
// finds even integer not less than n, with prime factors no larger than 5
// (ie, "smooth") and is a multiple of b (b is a number that the only prime
// factors are 2,3,5). Adapted from fortran in hellskitchen. Barnett 2/9/17
// changed INT64 type 3/28/17. Runtime is around n*1e-11 sec for big n.
// added condition about b Melody 05/31/20
{
  if (n <= 2)
    return 2;
  if (n % 2 == 1)
    n += 1;          // even
  int nplus = n - 2; // to cancel out the +=2 at start of loop
  int numdiv = 2;    // a dummy that is >1
  while ((numdiv > 1) || (nplus % b != 0))
  {
    nplus += 2; // stays even
    numdiv = nplus;
    while (numdiv % 2 == 0)
      numdiv /= 2; // remove all factors of 2,3,5...
    while (numdiv % 3 == 0)
      numdiv /= 3;
    while (numdiv % 5 == 0)
      numdiv /= 5;
  }
  return nplus;
}

void show_mem_usage()
{
  // show memory usage of GPU

  size_t free_byte;

  size_t total_byte;

  CHECK(cudaMemGetInfo(&free_byte, &total_byte));

  double free_db = (double)free_byte;

  double total_db = (double)total_byte;

  double used_db = total_db - free_db;

  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
         used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}