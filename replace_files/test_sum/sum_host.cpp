#include "sum_host.hpp"
#include <string>
#include <unordered_set>
#include <bsg_manycore_regression.h>
#define ALLOC_NAME "default_allocator"

typedef float data_t;

using namespace std;

int reference_vec_sum(int *vec, int vec_length) {
  int result = 0;
  for (int i = 0; i < vec_length; i++) {
    result += vec[i];
    vec[i]=0;
    }
  vec[0]=result;
  return 0;
}


int check_result_correctness(int *A, int *B) {
    int result = 0; 
    result += 256 - B[0];
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


  int vec_length = 256;
  int vec_size= sizeof(int) * vec_length;
 
 
  int* vec_cpy = (int*)malloc(vec_length * sizeof(int));
  int* host_vec = (int*)malloc(vec_length * sizeof(int));
  int* hb_vec = (int*)malloc(vec_length * sizeof(int));
  for (int i = 0; i < vec_length ; i++) {
      vec_cpy[i] = 1;
      host_vec[i] = 1;
      hb_vec[i] = 0;
  }

  for (int i =0; i < 5; i++) {
    printf("vector to be copied is is %d\n", vec_cpy[i]);
}



reference_vec_sum(host_vec, vec_length);

/*-----------------------------------------------------------------------------------------------------*/

//copy the vector to HB//

 eva_t vec_cpy_dev;
 eva_t hb_vec_dev;
  
 err  = hb_mc_device_malloc(&device, vec_size, &vec_cpy_dev);
 if (err != HB_MC_SUCCESS) {
   printf("failed to allocate memory on device for vec_cpy\n");
  }
 printf("start copying vec_cpy to device\n");
 hb_mc_dma_htod_t vec_cpy_dma = {vec_cpy_dev, (void*)(vec_cpy), vec_size};
 err |= hb_mc_device_dma_to_device(&device, &vec_cpy_dma, 1);


err  = hb_mc_device_malloc(&device, vec_size, &hb_vec_dev);
 if (err != HB_MC_SUCCESS) {
   printf("failed to allocate memory on device for hb_vec.\n");
  }
 printf("start copying hb_vec to device\n");
 hb_mc_dma_htod_t hb_vec_dma = {hb_vec_dev, (void*)(hb_vec), vec_size};
 err |= hb_mc_device_dma_to_device(&device, &hb_vec_dma, 1);

  hb_mc_dimension_t grid_dim = { .x = 0, .y = 0};
  hb_mc_dimension_t tg_dim = { .x = 0, .y = 0 };
  hb_mc_dimension_t block_size = { .x = 0, .y = 0 };

  grid_dim = { .x = 1, .y = 1};
  tg_dim = { .x = 16, .y = 8 };

/*-----------------------------------------------------------------------------------------------------*/

//prepare input arguments for kernel

  uint32_t cuda_argv[3] = {vec_cpy_dev, hb_vec_dev, vec_length};
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
  err |= hb_mc_device_memcpy (&device, (void*)(hb_vec), (void*)((intptr_t)hb_vec_dev),
                         vec_size, HB_MC_MEMCPY_TO_HOST);
  if (err != HB_MC_SUCCESS) {
    printf("ERROR: failed to copy vector to host\n");
  }

  for (int i =0; i < 3; i++) {
    printf("hb_vec is %d.\n", hb_vec[i]);
  }

  int error;

  printf("check_dense.\n");
  error = check_result_correctness(host_vec, hb_vec);
  printf("Result is %d\n", error );

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

  free(vec_cpy);
  free(hb_vec);

  return HB_MC_SUCCESS;
}

declare_program_main("test_sum", test_sum);
