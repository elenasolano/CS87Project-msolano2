// CS 87 - Final Project
// Maria-Elena Solano
//
// This utility simply counts how many CPU cores are in this node
//

#include <stdio.h>                     // C's standard I/O library
#include <stdlib.h>                    // C's standard library
#include <stdint.h>                    // C's exact width int types
#include <unistd.h>                    // C's POSIX API
#include <cuda_runtime.h>              // CUDA runtime library


// macro/constant definitions
#define cuda_try(X)                    ((X) != cudaSuccess)
#define perror_out(X)                  perror(X), fflush(stderr)
#define stderr_out(...)                fprintf(stderr, __VA_ARGS__), \
                                        fflush(stderr)
#define print_out(...)                 printf(__VA_ARGS__), fflush(stdout)


// simple helper container (used in how_many_warp_schedulers_per_SM)
typedef struct sm_to_ws{
  int sm;
  int ws;
} sm_to_ws_t;


void print_count();
void print_count_verbose();
int  how_many_warp_schedulers_per_SM(int arch_major_ver, int arch_minor_ver);


int main(int argc, char** argv){
  int ret;                                  // return value from getopt, and

  // Greedily read all the command line options provided.
  while((ret = getopt(argc, argv, "v")) != -1){
    switch(ret){
      // If option -v, display the full calculation instead
      case 'v':{
        goto verbose;
      }
    }
  }

  // Print the number of cores
  print_count();

  // And return
  goto done;


verbose:
  // If 'verbose' mode, display the model, and the full calculation
  print_count_verbose();  

done:
  exit(EXIT_SUCCESS);
}


// This function prints out the number of independent blocks in the GPU.
//
void print_count(){
  cudaError_t ret;                     // return value of CUDA calls
  int dev, devs;                       // number of devices
  cudaDeviceProp pr;                   // device properties
  uint32_t p;                          // number of concurrent blocks

  // Count how many devices are there. If err or no devices, set p=0 and print.
  if(cuda_try(ret = cudaGetDeviceCount(&devs)) || devs == 0){
    stderr_out("cudaGetDeviceCount error: %s\n", cudaGetErrorString(ret));
    p = 0;
    goto print;
  }

  // Get the device properties of the last device
  dev = devs - 1;
  cudaSetDevice(dev);
  cudaGetDeviceProperties(&pr, dev);

  // Compute the number of concurrent blocks according to the formula 
  //
  //   # of SMs x # of warp schedulers per SM
  //
  p=pr.multiProcessorCount*how_many_warp_schedulers_per_SM(pr.major,pr.minor);

print:
  print_out("%u\n", p);

  return;
}


// This function prints out the number of independent blocks in the GPU,
// showing how the calculation was made: that is, # of SMs x # of warp sche-
// dulers per SM available in the given model.
//
void print_count_verbose(){
  cudaError_t ret;                     // return value of CUDA calls
  int dev, devs;                       // number of devices
  cudaDeviceProp pr;                   // device properties

  // Count how many devices are there. If err, return.
  if(cuda_try(ret = cudaGetDeviceCount(&devs))){
    stderr_out("cudaGetDeviceCount error: %s\n", cudaGetErrorString(ret));
    return;
  }

  // If no devices, notify the user and return
  if(devs == 0){
    print_out("  (No CUDA-enabled GPUs in this machine.)\n");
    return;
  }

  // Get the device properties of the last device
  dev = devs - 1;
  cudaSetDevice(dev);
  cudaGetDeviceProperties(&pr, dev);

  // Show the calculation
  //
  //   # of SMs x # of warp schedulers per SM
  //
  print_out("  (%u SMs x %u warp schedulers per SM)\n",
    pr.multiProcessorCount,
    how_many_warp_schedulers_per_SM(pr.major, pr.minor));

  return;
}

// This function determines how many warp schedulers per SM are there in the
// GPU given ts major and minor architectural version.
// 
// Adapted from helper_cuda.h in the CUDA SDK:
//   (/usr/local/cuda/samples/common/inc/helper_cuda.h).
//
//   major, minor:                     major and minor architecture version
//
//   returns:                          number of warp schedulers per SM
//
int how_many_warp_schedulers_per_SM(int major, int minor){
  int i;
  sm_to_ws_t t[13];                   // Lookup table

  // Tesla architecture (1 warp scheduler per SM)
  t[0] .sm = 0x10;  t[0] .ws = 1;     // Tesla   (SM 1.0) G80 class
  t[1] .sm = 0x11;  t[1] .ws = 1;     // Tesla   (SM 1.1) G8X class
  t[2] .sm = 0x12;  t[2] .ws = 1;     // Tesla   (SM 1.2) G9X class
  t[3] .sm = 0x13;  t[3] .ws = 1;     // Tesla   (SM 1.3) GT200 class

  // Fermi architecture (2 warp schedulers per SM)
  t[4] .sm = 0x20;  t[4] .ws = 2;     // Fermi   (SM 2.0) GF100 class
  t[5] .sm = 0x21;  t[5] .ws = 2;     // Fermi   (SM 2.1) GF10x class

  // Kepler architecture (4 warp schedulers per SM)
  t[6] .sm = 0x30;  t[6] .ws = 4;     // Kepler  (SM 3.0) GK10x class
  t[7] .sm = 0x32;  t[7] .ws = 4;     // Kepler  (SM 3.2) GK10x class
  t[8] .sm = 0x35;  t[8] .ws = 4;     // Kepler  (SM 3.5) GK11x class
  t[9] .sm = 0x37;  t[9] .ws = 4;     // Kepler  (SM 3.7) GK21x class

  // Maxwell architecture (4 warp schedulers per SM)
  t[10].sm = 0x50;  t[10].ws = 4;     // Maxwell (SM 5.0) GM10x class
  t[11].sm = 0x52;  t[11].ws = 4;     // Maxwell (SM 5.2) GM20x class

  // Unknown architecture
  t[12].sm = -1;    t[12].ws = -1;    // Unknown

  for(i=0; i<13; i++){
    if(t[i].sm == ((major << 4) + minor)){
      return t[i].ws;
    }
  }
  return 0;
}
