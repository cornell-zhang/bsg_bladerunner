#include "sum_host.h"
#include <string.h>
//#include <unordered_set>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_errno.h>
#include <bsg_manycore_tile.h>
#include <bsg_manycore_loader.h>
#include <bsg_manycore_cuda.h>
#include <bsg_manycore_regression.h>
#define ALLOC_NAME "default_allocator"

typedef float data_t;

//using namespace std;

void reference_vec_sum(int *host_vec, int* host_result, int input_length) {
  int result = 0;
  for (int i = 0; i < input_length; i++) {
    result += host_vec[i];
  }
  host_result[128] = result;
}


int check_result_correctness(int *A, int *B) {
  int result;
  result = A[128] - B[128];
  return result;
}

int test_sum (int argc, char *argv[]) {

  char *bin_path, *test_name;
  struct arguments_path args = {NULL, NULL};
  argp_parse(&argp_path, argc, argv, 0, 0, &args);
  bin_path = args.path;
  test_name = args.name;


  printf("start creating device\n");
  hb_mc_device_t device;
  int err;
  err = hb_mc_device_init(&device, test_name, 0);
  if (err != HB_MC_SUCCESS) {
    printf("failed to initialize device.\n");
    return err;
  }

  printf("start doing hb_mc_device_program_init\n");
  err = hb_mc_device_program_init(&device, bin_path, "default_allocator", 0);
  if (err != HB_MC_SUCCESS) {
    printf("failed to initialize program.\n");
    return err;
  }


  int input_length = 3096;
  int output_length = 129;
 
  int* host_vec = (int*)malloc(input_length * sizeof(int));
  int* hb_vec = (int*)malloc(input_length * sizeof(int));
  int* host_result = (int*)malloc(output_length * sizeof(int));
  int* hb_result = (int*)malloc(output_length * sizeof(int));
  for (int i = 0; i < input_length ; i++) {
      host_vec[i] = 1;
      hb_vec[i] = 1;
  }

  for (int i =0; i < 5; i++) {
    printf("vector to be copied is is %d\n", host_vec[i]);
  }



  reference_vec_sum(host_vec, host_result, input_length);

/*-----------------------------------------------------------------------------------------------------*/

//copy the vector to HB//

  eva_t hb_vec_dev;
  eva_t hb_result_dev;
  
  err  = hb_mc_device_malloc(&device, input_length * sizeof(int), &hb_vec_dev);
  if (err != HB_MC_SUCCESS) {
    printf("failed to allocate memory on device for vec_cpy\n");
  }
  printf("start copying vec_cpy to device\n");
  hb_mc_dma_htod_t hb_vec_dma = {hb_vec_dev, (void*)(hb_vec), input_length * sizeof(int)};
  err |= hb_mc_device_dma_to_device(&device, &hb_vec_dma, 1);


  err  = hb_mc_device_malloc(&device, output_length, &hb_result_dev);
  if (err != HB_MC_SUCCESS) {
    printf("failed to allocate memory on device for hb_vec.\n");
  }
  printf("start copying hb_vec to device\n");
  hb_mc_dma_htod_t hb_result_dma = {hb_result_dev, (void*)(hb_result), output_length * sizeof(int)};
  err |= hb_mc_device_dma_to_device(&device, &hb_result_dma, 1);

  hb_mc_dimension_t grid_dim = { .x = 1, .y = 1};
  hb_mc_dimension_t tg_dim = { .x = 16, .y = 8 };
  hb_mc_dimension_t block_size = { .x = 0, .y = 0 };

//  grid_dim = { .x = 1, .y = 1};
//  tg_dim = { .x = 16, .y = 8 };

/*-----------------------------------------------------------------------------------------------------*/

//prepare input arguments for kernel

  uint32_t cuda_argv[3] = {hb_vec_dev, hb_result_dev, input_length};
  int cuda_argc = 3;

  printf("hb_mc_kernel_enqueue\n");
  err = hb_mc_kernel_enqueue(&device, grid_dim, tg_dim, "kernel_sum", cuda_argc, cuda_argv);

  if (err != HB_MC_SUCCESS) {
    printf("failed to hb_mc_kernel_enqueue.\n");
    return err;
  }

  printf("hb_mc_device_tile_groups_execute\n");
  err = hb_mc_device_tile_groups_execute(&device);
  if (err != HB_MC_SUCCESS) {
    printf("failed to execute tile groups.\n");
    return err;
  }

/*-----------------------------------------------------------------------------------------------------*/

//copy result from device to host//

  printf("Copy device to Host.\n");
  err |= hb_mc_device_memcpy (&device, (void*)(hb_result), (void*)((intptr_t)hb_result_dev),
                         output_length * sizeof(int), HB_MC_MEMCPY_TO_HOST);
  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy vector to host\n");
  }

  printf("hb_result is %d.\n", hb_result[128]);

  int error;

  error = check_result_correctness(host_result, hb_result);
  printf("Error is %d\n", error );

  if (error != 0) {
    bsg_pr_test_err(BSG_RED("Mismatch. Error: %d\n"), error);
    return HB_MC_FAIL;
  }
  bsg_pr_test_info(BSG_GREEN("Match.\n"));

/*-----------------------------------------------------------------------------------------------------*/
	
/*freeze Tiles */

 err = hb_mc_device_finish(&device);
  if (err != HB_MC_SUCCESS) {
    printf("failed to de-initialize device.\n");
    return err;
  }

  free(hb_vec);
  free(hb_result);
  free(host_result);
  free(host_vec);

  return HB_MC_SUCCESS;
}

declare_program_main("test_sum", test_sum);
