//this is the kernel code for vector_sum 
//main goal of this assignment is to initalize a vector on the host, transfer it from host to device, add all the elements and store the result, send it back to host 

#define BSG_TILE_GROUP_X_DIM 16
#define BSG_TILE_GROUP_Y_DIM 8
#define bsg_tiles_X BSG_TILE_GROUP_X_DIM
#define bsg_tiles_Y BSG_TILE_GROUP_Y_DIM
#include <bsg_manycore.h>
#include <bsg_set_tile_x_y.h>
#include <bsg_tile_group_barrier.hpp>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <stdio.h>

bsg_barrier<bsg_tiles_X, bsg_tiles_Y> barrier;

extern int bsg_printf(const char*, ...);

extern void* g_reduction_buffer = NULL;

extern "C" {


int  __attribute__ ((noinline)) kernel_sum(int *vec_cpy,
					   int *hb_vec,
					   int vec_length) { 
 int thread_num =bsg_tiles_X * bsg_tiles_Y;
 bsg_cuda_print_stat_kernel_start();

 //vec_length = 256
 //thread_num = 128

 int* buffer = (int*) g_reduction_buffer;

 for (int j = __bsg_id ; j < vec_length ; j += thread_num) 
  buffer[__bsg_id] += vec_cpy[j]; 

 barrier.sync();

 if (__bsg_id == 0) 
  for ( int i = 0; i < thread_num; i++) 
    hb_vec[0] += buffer[i];
  

 bsg_cuda_print_stat_kernel_end();
 barrier.sync();

 return 0;

 } 

}    
