#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
//#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "dataType.h"

///contrib

// 0 correct  1 warning 2 error

#define CHECK(call)                                               \
  {                                                               \
    const cudaError_t error = call;                               \
    if (error != cudaSuccess)                                     \
    {                                                             \
      printf("Error:%s:%d", __FILE__, __LINE__);                  \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString); \
      exit(1);                                                    \
    }                                                             \
  }

#define M_1_2PI 0.159154943091895336 // 1/2/pi for faster calculation
#define M_2PI 6.28318530717958648    // 2 pi

#define PI (PCS) M_PI

#define EPSILON (double)1.1e-16

#define BLOCKSIZE 16

#define SPEEDOFLIGHT 299792458.0

#define MAX_CUFFT_ELEM 128e6

//random11 and rand01
// Random numbers: crappy unif random number generator in [0,1):
//#define rand01() (((PCS)(rand()%RAND_MAX))/RAND_MAX)
#define rand01() ((PCS)rand() / RAND_MAX)
// unif[-1,1):
#define randm11() (2 * rand01() - (PCS)1.0)
void rescaling_real_invoker(PCS *d_x, PCS scale_ratio, int N);
void rescaling_complex_invoker(CUCPX *d_x, PCS scale_ratio, int N);
int next235beven(int n, int b);
void prefix_scan(PCS *d_arr, PCS *d_res, int n, int flag);
void get_max_min(PCS &max, PCS &min, PCS *d_array, int n);
int matrix_transpose_invoker(PCS *d_arr, int width, int height);
void GPU_info();
void show_mem_usage();
int matrix_elementwise_multiply_invoker(CUCPX *a, PCS *b, int N);
int matrix_elementwise_divide_invoker(CUCPX *a, PCS *b, int N);
#endif