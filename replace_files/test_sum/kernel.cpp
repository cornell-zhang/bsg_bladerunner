//this is the kernel code for vector_sum 
//main goal of this assignment is to initalize a vector on the host, transfer it from host to device, add all the elements and store the result, send it back to host 

#define BSG_TILE_GROUP_X_DIM 16
#define BSG_TILE_GROUP_Y_DIM 8
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include "bsg_manycore.h"
#include "bsg_set_tile_x_y.h"
#include "bsg_cuda_lite_barrier.h"
#include "bsg_barrier_amoadd.h"
//#include <bsg_tile_group_barrier.hpp>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdio.h>

//bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

extern int bsg_printf(const char*, ...);

extern void* g_reduction_buffer = NULL;

extern void bsg_barrier_amoadd(int*, int*);
extern "C" {


int  __attribute__ ((noinline)) kernel_sum(int *hb_vec,
					   int *hb_result,
					   int vec_length) { 
  int thread_num =bsg_tiles_X * bsg_tiles_Y;
  bsg_barrier_hw_tile_group_init();
  bsg_cuda_print_stat_kernel_start();

  int* buffer = (int*) g_reduction_buffer;
  int temp_buff = 0;
  for (int j = __bsg_id ; j < vec_length ; j += thread_num) {
    temp_buff += hb_vec[j];
  }
  hb_result[__bsg_id] = temp_buff;

  bsg_barrier_hw_tile_group_sync();

  if (__bsg_id == 0) {
    int result = 0;
    for (unsigned i = 0; i < thread_num; i++) { 
      result += hb_result[i];
    }
    hb_result[thread_num] = result;
  }
  
  bsg_cuda_print_stat_kernel_end();
  bsg_barrier_hw_tile_group_sync();

 return 0;

 } 

}    
