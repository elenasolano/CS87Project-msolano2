// CS 87 - Final Project
// Maria-Elena Solano
//
// Radix-2 Cooley-Tukey Fourier Transform on C^n - parallel 'pi' CUDA version
//

#include <stdio.h>                     // C's standard I/O library
#include <stdlib.h>                    // C's standard library
#include <stdint.h>                    // C's exact width int types
#include <string.h>                    // C's standard string library
#include <time.h>                      // C's time types (for random init)
#include <unistd.h>                    // C's POSIX API
#include <cuda.h>                      // CUDA runtime API


// macro/constant definitions
#define perror_out(X)                  perror(X), fflush(stderr)
#define stderr_out(...)                fprintf(stderr, __VA_ARGS__), \
                                        fflush(stderr)
#define print_out(...)                 printf(__VA_ARGS__), fflush(stdout)
#define cuda_try(X)                    ((X) != cudaSuccess)
#define cuda_malloc(X, Y)              cudaMalloc((void**)(X), (Y))
#define cuda_memcpy_todev(X, Y, Z)     cudaMemcpy((X), (Y), (Z), \
                                        cudaMemcpyHostToDevice)
#define cuda_memcpy_tohost(X, Y, Z)    cudaMemcpy((X), (Y), (Z), \
                                        cudaMemcpyDeviceToHost)
                                        

// data structures
// unit data type
typedef struct complex{                
  float re;                            // real part
  float im;                            // imaginary part
} data_t;

// simple struct to hold transform info
typedef struct tr tr_t;
typedef struct tr{ 
  uint32_t  N;                         // input size
  uint32_t  P;                         // number of processors
  uint32_t  Pi;                        // processor ID
  data_t*   in;                        // input data,
  data_t*   out;                       // output,
  data_t*   tmp_in;                    // input scratchpad,
  data_t*   tmp_out;                   // output scratchpads, and
  uint32_t  test_mode;                 // test mode?
  uint32_t  no_header;                 // no timing headers?
  tr_t*     trs;                       // array of tr_t objects in host,
  tr_t*     trs_dev;                   // and device.
} tr_t;

// simple encapsulating struct for timing
typedef struct timer{
  cudaEvent_t start, stop;             // start and stop events
  float  elapsed;                      // elapsed time (if any), in millisecs
} tmr_t;

// simple helper container (used in how_many_warp_schedulers)
typedef struct sm_to_ws{
  int sm;
  int ws;
} sm_to_ws_t;


// function declarations
// setting up the transform
int  setup_from_args (tr_t* t, int argc, char** argv);
void show_usage      ();

// running the transform
int   run             (tr_t* t);
int   initialize_data (tr_t* t);
void  cleanup_data    (tr_t* t);

__global__ void run_funnel_stage(tr_t* t);
__global__ void run_tube_stage  (tr_t* t);
__device__ void butterfly     (data_t* out, data_t* in, 
                               uint32_t size, uint32_t N);
__device__ void butterfly_half(data_t* out, data_t* in, 
                               uint32_t size, uint32_t N, uint32_t which_half);

// complex arithmetic
__device__ data_t add  (data_t a, data_t b);
__device__ data_t sub  (data_t a, data_t b);
__device__ data_t mul  (data_t a, data_t b);
__device__ data_t omega(uint32_t N, uint32_t k);
__device__ data_t convex_comb(data_t a, data_t b, uint32_t alpha);

// printing and verifying
void print_input     (tr_t* t);
void print_output    (tr_t* t);
void verify_results  (tr_t* t);

// timing
void timer_start(tmr_t* tm);
void timer_stop (tmr_t* tm);

// misc
__device__ __host__ uint32_t ilog2 (uint32_t x);
__device__   void swap_scratchpads (tr_t* t);
int      is_power_of_two           (int x);
uint32_t hash                      (uint32_t x);
uint32_t bit_reverse               (uint32_t x, uint32_t N);
uint32_t how_many_concurrent_blocks();
int      how_many_warp_schedulers  (int arch_major_ver, int arch_minor_ver);


int main(int argc, char** argv){
  tr_t t;                              // main transform object

  // (1) setup the transform from the command line args (or ret -1 if err)
  if(setup_from_args(&t, argc, argv)){
    exit(EXIT_FAILURE);
  }

  // (2) run the transform according to the given args (or ret -1 if err)
  if(run(&t)){
    exit(EXIT_FAILURE);    
  }

  // (3) return
  exit(EXIT_SUCCESS);
}


// This function parses the provided command line args into the given transform
// object, and initializes it accordingly. Returns 0 if success, or -1 if err).
//
//   t:                      (ptr to) tr_t object to update
//   argc:                   number of command line arguments
//   argv:                   array of strings containing the command line args
//
//   returns:                0 if success, or -1 if invalid command line args.
// 
int setup_from_args(tr_t* t, int argc, char** argv){
  int ret;                                  // return value from getopt, and
  int num = 0;                              // number to be retrieved.

  // First, zero all the entries of the tr_t (so we can tell whether or not
  // the entries were filled afterwards simply by checking if they're still 0)
  memset(t, 0, sizeof(tr_t));

  // Then, greedily read all the command line options provided.
  while((ret = getopt(argc, argv, "n:p:to")) != -1){
    switch(ret){
      // If option -n, grab the arg, which should be a nonnegative power of 2
      // (otherwise return -1 err)
      case 'n':{
        if(!(num = atoi(optarg)) || !(num > 1) || !is_power_of_two(num)){
          stderr_out("Invalid input size (should be 2^i for i>0)\n");
          show_usage(); 
          goto err;
        }
        t->N = num;
        break;
      }
      // If option -p, grab the arg, which should be 1 or a nonneg power of 2
      // (otherwise return -1 err)
      case 'p':{
        if(!(num = atoi(optarg)) || !(num > 0) || !is_power_of_two(num)){
          stderr_out("Invalid number of procs (should be 2^i for i>0)\n");
          show_usage();
          goto err;
        }
        t->P = num;
        break;
      }
      // If option -o, set t.no_header to 1.
      case 'o':{
        t->no_header = 1;
        break;
      }
      // If option -t, set t.N to 8 and t.test_mode to 'true', and notify user.
      case 't':{
        print_out("Test mode (ignoring provided input size, if any)...\n");
        t->N = 8;
        t->test_mode = 1;
        break;
      }
      // if unknown or missing arg, show usage and return -1 (error)
      case '?':{
        stderr_out("Unknown or missing arg %c\n", optopt);
        show_usage();
        goto err;
      }
    }
  }

  // Finally, validate the options
  // If the -n option is missing, notify the user, show usage, and return -1.
  if(!t->N){
    stderr_out("Missing option: -n\n");
    show_usage();
    goto err;
  }
  // If the -p option is missing, notify the user, show usage, and return -1.
  if(!t->P){
    stderr_out("Missing option: -p\n");
    show_usage();
    goto err;
  }
  // If more processors than inputs, return -1.
  if(t->P > t->N){
    stderr_out("More processors than inputs!\n");
    show_usage();
    goto err;
  }
  // If more processors than available, return -1.
  if(t->P > how_many_concurrent_blocks()){
    stderr_out("Too many processors! (only %u concurrent blocks available)\n",
               how_many_concurrent_blocks());
    goto err;
  }

  return 0;

err:
  stderr_out("Could not setup the transform from the cmdline args\n");
  return -1;
}

// This function initializes the given transform's data. Returns 0 if success 
// or -1 if err.
//
//   t:                      (ptr to) tr_t object to update
//
//   returns:                0 if success, or -1 if error.
// 
int initialize_data(tr_t* t){
  uint32_t bit;
  cudaError_t ret;                          // return value of CUDA calls

  // We need:
  // - N entries for the input,
  if((t->in = (data_t*)malloc(sizeof(data_t)*t->N)) == NULL){
    perror_out("malloc error");
    goto err;
  }
  // - N entries for the output, and
  if((t->out = (data_t*)malloc(sizeof(data_t)*t->N)) == NULL){
    perror_out("malloc error");
    goto err;
  }
  // - P tr_t objects (one for each block) (on both the host and the device),
  if((t->trs = (tr_t*)malloc(sizeof(tr_t)*t->P)) == NULL){
    perror_out("malloc error");
    goto err;
  }
  if(cuda_try(ret = cuda_malloc(&t->trs_dev, sizeof(tr_t)*t->P))){
    stderr_out("cudaMalloc error: %s\n", cudaGetErrorString(ret));
    goto err;
  }
  // - And, for each processor,
  for(t->Pi = 0; t->Pi < t->P; t->Pi++){
    
    // (Initialize their respective tr_t objects first)
    t->trs[t->Pi] = *t;

    // An input scratchpad of size N, and
    if(cuda_try(ret = cuda_malloc(&t->trs[t->Pi].tmp_in, 
                                  sizeof(data_t)*t->N))){
      stderr_out("cudaMalloc error: %s\n", cudaGetErrorString(ret));
      goto err;
    }

    // An output scratchpad of size N.
    if(cuda_try(ret = cuda_malloc(&t->trs[t->Pi].tmp_out, 
                                  sizeof(data_t)*t->N))){
      stderr_out("cudaMalloc error: %s\n", cudaGetErrorString(ret));
      goto err;
    }

    // (Making sure to copy the updated tr_t to the device's memory too)
    if(cuda_try(ret = cuda_memcpy_todev(t->trs_dev + t->Pi, 
                                        t->trs     + t->Pi,
                                        sizeof(tr_t)))){
      stderr_out("cudaMemcpy error: %s\n", cudaGetErrorString(ret));
      goto err;
    }

  }

  
  // Initialize the input with Bernoulli deviates from the L2 unit circle
  // --i.e. each element drawn from the distribution 1/sqrt{N}*[-1,1].
  if(!t->test_mode){
    for(srand(hash(time(NULL))), bit = 0; bit < t->N; bit++){
      t->in[bit].re = (rand()/(RAND_MAX/2.0)-1.0)/sqrt(t->N);
      t->in[bit].im = (rand()/(RAND_MAX/2.0)-1.0)/sqrt(t->N);
    }
  }
  // (unless in debug mode, in which case the values are always the same
  //  test case: 0,1,0,1,0,1,0,1 (the output should be 4,0,0,0,-4,0,0,0))
  else{
    t->in[0].re = 0; t->in[0].im = 0;
    t->in[1].re = 1; t->in[1].im = 0;
    t->in[2].re = 0; t->in[2].im = 0;
    t->in[3].re = 1; t->in[3].im = 0;
    t->in[4].re = 0; t->in[4].im = 0;
    t->in[5].re = 1; t->in[5].im = 0;
    t->in[6].re = 0; t->in[6].im = 0;
    t->in[7].re = 1; t->in[7].im = 0;
    print_input(t);
  }

  // - And copy the input to the input scratchpad of each block (of size N).
  for(t->Pi = 0; t->Pi < t->P; t->Pi++){
    if(cuda_try(ret = cuda_memcpy_todev(t->trs[t->Pi].tmp_in,
                                        t->in, sizeof(data_t)*t->N))){
      stderr_out("cudaMemcpy error: %s\n", cudaGetErrorString(ret));
      goto err;
    }
  }

  // And return
  return 0;

err:
  stderr_out("Could not initialize the data\n");
  return -1;
}

// This function frees all the data allocated by the given transform. 
//
//   t:                      (ptr to) tr_t object to update
// 
void cleanup_data(tr_t* t){
  cudaError_t ret;                          // return value of CUDA calls

  // Free all the allocated data (if any)
  if(t->in  != NULL){ free(t->in);   t->in  = NULL; }
  if(t->out != NULL){ free(t->out);  t->out = NULL; }
  if(t->trs != NULL){ 
    for(t->Pi = 0; t->Pi < t->P; t->Pi++){
      if(t->trs[t->Pi].tmp_in != NULL && 
         cuda_try(ret = cudaFree(t->trs[t->Pi].tmp_in))){
        stderr_out("cudaFree error: %s\n", cudaGetErrorString(ret)); };
      if(t->trs[t->Pi].tmp_out != NULL && 
         cuda_try(ret = cudaFree(t->trs[t->Pi].tmp_out))){
        stderr_out("cudaFree error: %s\n", cudaGetErrorString(ret)); };
    }  
    free(t->trs);  t->trs = NULL; 
  }
  if(t->trs_dev != NULL && cuda_try(ret = cudaFree(t->trs_dev))){
    stderr_out("cudaFree error: %s\n", cudaGetErrorString(ret)); 
  }

  return;
}

// This function shows the cmdline interface usage.
// 
void show_usage(){
  print_out("\nusage:\n"
            "  fourier-parallel-pi-gpu-cuda { -n <n> -p <p> [-o] | -t }\n"
            "\noptions:\n"
            "  -n <n>     power of two input size\n"
            "  -p <p>     power of two number of processors (less than n)\n"
            "  -o         omit timing headers\n"
            "  -t         compare against precomputed input/output\n"
            "\n");
}


// This function runs the given transform on P separate processors. Returns 0
// if success or -1 if err.
//
//   t:                      (ptr to) tr_t object to update
//
//   returns:                0 if success, or -1 if error.
// 
int run(tr_t* t){
  tmr_t       tm_funnel, tm_tube;           // tmr_t objects (for timing)
  cudaError_t ret;                          // return value of CUDA calls
  uint32_t    bit;

  // Initialize the data (or ret -1 err, cleaning up any malloc'd data first)
  if(initialize_data(t)){
    cleanup_data(t);
    goto err;
  }

  // Run the 1st (aka 'funnel') stage on P blocks, 1 thread per block.
  timer_start(&tm_funnel);
  run_funnel_stage<<<t->P, 1>>>(t->trs_dev);
  timer_stop(&tm_funnel);

  // Run the 2nd (aka 'tube') stage on P blocks, 1 thread per block.
  timer_start(&tm_tube);
  run_tube_stage<<<t->P, 1>>>(t->trs_dev);
  timer_stop(&tm_tube);

  // If not in test mode, show elapsed times (total, tree and cylinder stages)
  if(!t->test_mode){
    if(!t->no_header){ 
      print_out("n\tp\ttime (total)\t"
                "time (stage 1)\ttime (stage 2)\n"); };
    print_out("%u\t%u\t%f\t%f\t%f\n", 
              t->N, t->P, tm_funnel.elapsed + tm_tube.elapsed, 
              tm_funnel.elapsed, tm_tube.elapsed);
  }
  // Otherwise, copy the result from the _assigned segment_ (of size N/P) of
  // the input scratchpad of each processor (which will contain the new results
  // at this point) of each processor to the output, in bit-reversed order.
  //
  // Then print the output, and verify the results.
  else{
    // Update the host's tr_t array first
    if(cuda_try(ret = cuda_memcpy_tohost(
        t->trs, t->trs_dev, sizeof(tr_t)*t->P))){
      stderr_out("cudaMemcpy error: %s\n", cudaGetErrorString(ret));
    }
    for(t->Pi = 0; t->Pi < t->P; t->Pi++){
      for(bit = (t->N/t->P)*t->Pi; bit < (t->N/t->P)*t->Pi+t->N/t->P; bit++){
        if(cuda_try(ret = cuda_memcpy_tohost(
            t->out + bit_reverse(bit, ilog2(t->N)),   // position in output,
            t->trs[t->Pi].tmp_in + bit,               // Pi-th scratchpad.
            sizeof(data_t)))){
          stderr_out("cudaMemcpy error: %s\n", cudaGetErrorString(ret));
        }
      }
    }
    print_output(t);
    verify_results(t);
  }

  // Cleanup the data, and return
  cleanup_data(t);
  return 0;

err:
  stderr_out("Could not run the transform\n");
  return -1;
}

// This function runs the first stage of the transform in the GPU.
//
//   t_ptr:                  (ptr to) tr_t object to use
//
//   returns:                NULL always.
// 
__global__ void run_funnel_stage(tr_t* trs){
  uint32_t size, iter, offset;              // butterfly size, offset and iter,
  uint32_t which_butterfly, which_half;     // which butterfly, and which half,
  tr_t* t = trs + blockIdx.x;               // local, unpacked tr_t object

  // For the first log P iters
  // (i.e. butterfly sizes N,     N/2,       ..., 2(N/P),
  //       iteration stage log P, log P - 1, ..., 1),
  for(size = t->N, iter = ilog2(t->P); size > t->N/t->P; size /= 2, iter--){

  	// Determine _which_ butterfly to compute, depending on (Pi >> iter)
  	which_butterfly = (t->Pi >> iter);

    // Determine _which_ segment_ of size 'size' to compute the butterfly upon
    offset = which_butterfly * size;

    // Determine _which half_, depending on the parity of the 1st, 2nd, ...
    // most significant bit of Pi.
    which_half = ((t->Pi >> (iter - 1)) % 2 == 0);

    // And compute the butterfly accordingly
    butterfly_half(
      t->tmp_out + offset,        // output to the _1st_ half of the scratchpad
      t->tmp_in  + offset,        // and reading from the input scratchpad.
      size, t->N, which_half);

    // And swap scratchpads, so that the new results become the input for
    // the next iteration.
    swap_scratchpads(t);
  }

}

// This function runs the second stage of the transform in the GPU.
//
//   t_ptr:                  (ptr to) tr_t object to use
//
//   returns:                NULL always.
// 
__global__ void run_tube_stage(tr_t* trs){
  uint32_t size, offset;                    // butterfly size and offset
  tr_t* t = trs + blockIdx.x;               // local tr_t object

  // For the last log N/P iters
  // (i.e. butterfly sizes N/P, (N/P)/2, ..., 2), 
  // we work only on the portion of the output assigned to this processor
  for(size = t->N/t->P; size > 1; size /= 2){

    // Compute 1, 2, ..., (N/P)/2 butterflies of size N/P, (N/P)/2, ..., 2 
    // over _consecutive_ intervals across the assigned segment (of size N/P),
    // (which recall starts from (t.N/t.P)).
    for(offset = (t->N/t->P)*t->Pi; 
        offset < (t->N/t->P)*t->Pi + t->N/t->P; 
        offset += size){
      
      butterfly(
        t->tmp_out + offset,       // output to the _assigned_ segment,
        t->tmp_in  + offset,       // and reading from the _assigned_ segment.
        size, t->N);
    }

    // And swap scratchpads, so that the new results become the input for
    // the next iteration.
    swap_scratchpads(t);
  }
}


// This function computes an n-point butterfly over the given input array,
// placing the results on the given output array.
//
//   out,in:                 output and input arrays
//   size:                   size of the butterfly
//   N:                      input size of the broader FFT
//
__device__ void butterfly(data_t* out, data_t* in, uint32_t size, uint32_t N){

  // Compute the left butterfly (note it only writes on the _1st_ half of out)
  butterfly_half(out, in, size, N, 1);

  // And the right butterfly (note it only writes on the _2nd_ half of out)
  butterfly_half(out, in, size, N, 0);

  return;
}

// This function computes either half of an n-point butterfly. Note it only
// writes to either the _first_ half of the output (if left butterfly), or the
// _second_ half of the output (if right butterfly).
//
//   out,in:                 output and input arrays
//   size:                   size of the butterfly
//   N:                      input size of the broader FFT
//   which_half:             which half to compute?
//
__device__ void butterfly_half(data_t* out, data_t* in, 
                               uint32_t size, uint32_t N, uint32_t which_half){
  uint32_t bit;
      
  // For each bit in the segment
  for(out += !which_half*(size/2), bit = 0; bit < size/2; bit++){
    out[bit] =                              //  set the current (output) bit to
      convex_comb(                          //   either:
        add(                                //    the sum of
          in[bit],                          //     the current (input) bit, and
          in[bit+size/2]),                  //     the (size/2)-next one,
        mul(                                //    or the multiplication of
          sub(                              //     the sum of the
            in[bit],                        //      the (stride)-prev bit, and
            in[bit+size/2]),                //      the next one
          omega(N,                          //     times the Nth root of unity,
                bit*(N/size))),             //      to the power of 0,1,2 first
        which_half);                        //     0,2,4 second, etc.
  }

  return;
}

// This function returns (by value) the sum of the two given complex numbers.
//
//   a,b:                    operands (x+yi,z+wi) (of type data_t)
//
//   returns:                the sum a+b = (x+z)+(y+w)i
// 
__device__ data_t add(data_t a, data_t b){
  data_t c;

  c.re = a.re + b.re;
  c.im = a.im + b.im;

  return c;
}

// This function returns (by value) the sum a+(-b) of the given complex nums.
//
// (Note: this can be overriden to provide support for more general transforms
//        on arbitrary fields)
//
//   a,b:                    operands (x+yi,z+wi) (of type data_t)
//
//   returns:                the sum a+(-b) = (x-z)+(y-w)i
// 
__device__ data_t sub(data_t a, data_t b){
  data_t c;

  c.re = a.re - b.re;
  c.im = a.im - b.im;

  return c;
}

// This function returns (by value) the mult a*b of the given complex numbers.
//
// (Note: this can be overriden to provide support for more general transforms
//        on arbitrary fields)
//
//   a,b:                    operands (x+yi,z+wi) (of type data_t)
//
//   returns:                their mult a*b = (xz-yw) + (xw-yz)i
// 
__device__ data_t mul(data_t a, data_t b){
  data_t c;

  c.re = a.re*b.re - a.im*b.im;
  c.im = a.re*b.im + a.im*b.re;

  return c;
}

// This function returns (by value) the primitive N-th root of unity of the
// complex field \mathbb{C} the power of k, which is given by
//
//   e^{-2\pi k/N} = (cos 2\pi/N - i sin 2\pi/N)^k   (by Euler's formula)
//                 = cos 2\pi/N*k - i sin 2\pi/N*k   (by De Moivre's formula)
//
// (Note: this can be overriden to provide support for more general transforms
//        on arbitrary fields)
//
//   N:                      Order of the cyclic group.
//   k:                      Power to raise the root of unity to.
//
//   returns:                N-th primitive root of unity of \mathbb{C}, raised
//                           to the power of k.
// 
__device__ data_t omega(uint32_t N, uint32_t k){
  data_t o;

  o.re =  __cosf(2.0f*3.141592f/N*k);
  o.im = -__sinf(2.0f*3.141592f/N*k);

  return o;
}

// This function returns (by value) the 'convex combination' of a and b, para-
// meterized by the scalar alpha.
//
// (Note: this can be overriden to provide support for more general transforms
//        on arbitrary fields)
//
//   a,b:                    Operands
//   alpha:                  Alpha (between 0 and 1)
//
//   returns:                alpha*a + (1-alpha)*b
// 
__device__ data_t convex_comb(data_t a, data_t b, uint32_t alpha){
  data_t o;

  o.re = alpha*a.re + (1U-alpha)*b.re;
  o.im = alpha*a.im + (1U-alpha)*b.im;

  return o;
}



// This function prints out the input of the transform.
//
//   t:                      (ptr to) tr_t object to use.
//
void print_input(tr_t* t){
  uint32_t bit;                             // entry to print out

  for(print_out("Input:\n"), bit = 0; bit < t->N; bit++){
    print_out("%.1f+%.1fi, ", t->in[bit].re, t->in[bit].im);
  }
  print_out("\n");

  return;
}

// This function prints out the output of the transform.
//
//   t:                      (ptr to) tr_t object to use.
//
void print_output(tr_t* t){
  uint32_t bit;                             // entry to print out

  for(print_out("Output:\n"), bit = 0; bit < t->N; bit++){
    print_out("%.1f+%.1fi, ", t->out[bit].re, t->out[bit].im);
  }
  print_out("\n");

  return;
}

// This function verifies that the output of the transform is correct.
//
//   t:                      (ptr to) tr_t object to use.
//
void verify_results(tr_t* t){

  // since the input is 0,1,0,1,0,1,0,1; the output should be 4,0,0,0,-4,0,0,0.
  print_out(
    (t->out[0].re == 4 && t->out[0].im == 0 &&
     t->out[1].re == 0 && t->out[1].im == 0 &&
     t->out[2].re == 0 && t->out[2].im == 0 &&
     t->out[3].re == 0 && t->out[3].im == 0 &&
     t->out[4].re ==-4 && t->out[4].im == 0 &&
     t->out[5].re == 0 && t->out[5].im == 0 &&
     t->out[6].re == 0 && t->out[6].im == 0 &&
     t->out[7].re == 0 && t->out[7].im == 0)
    ? "Output is correct. Test passed.\n\n" 
    : "Output is incorrect! Test failed.\n\n");

  return;
}


// This function swaps the input and output scratchpads of the given tr_t obj.
//
//   t:                      (ptr to) tr_t object to use.
//
__device__ void swap_scratchpads(tr_t* t){
  data_t* temp;

  temp       = t->tmp_in;  
  t->tmp_in  = t->tmp_out;  
  t->tmp_out = temp;

  return;
}

// This function determines whether or not the given number is a non-negative
// power of two.
//
// From: www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
//
//   x:                      value to query.
//
//   returns:                0 if _not_ a non-negative power of 2, 1 otherwise.
// 
int is_power_of_two(int x){
  return x && !(x & (x - 1));
}

// This function computes a hash of the given uint32 number.
// (From http://www.concentric.net/~ttwang/tech/inthash.htm)
//
//   x:                      number to hash.
//
//   returns:                uint32 hash of the given number.
// 
uint32_t hash(uint32_t x){
  x = (x + 0x7ed55d16U) + (x << 12);  x = (x ^ 0xc761c23cU) ^ (x >> 19);
  x = (x + 0x165667b1U) + (x << 5);   x = (x + 0xd3a2646cU) ^ (x << 9);
  x = (x + 0xfd7046c5U) + (x << 3);   x = (x ^ 0xb55a4f09U) ^ (x >> 16);
  return x;
}

// This function computes the log2 of the given uint32_t number.
//
// From: graphics.stanford.edu/~seander/bithacks.html#IntegerLogDeBruijn
//
//   x:                      value to compute.
//
//   returns:                log2(x)
// 
__host__ __device__ uint32_t ilog2(uint32_t x){  
  int lookup_table[32] = {
     0,  1, 28,  2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17,  4, 8, 
    31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18,  6, 11,  5, 10, 9 };

  return lookup_table[(uint32_t)(x*0x077CB531U) >> 27];
}


// This function bit-reverses the given index (represented as a m-bit binary 
// number), and returns the resulting value as an integer.
// 
// This is used in the final step of the FFT: due to the recursive nature of
// the factorization, the butterfly's outputs are always in bit-reversed order
// (when represented as log(N)-bit numbers, where N is the input size). So for
// instance for a butterfly of size 8, the 1st output of the butterfly actually
// maps to the 1 -> 001 -> 100 -> 4th output of the array; while the 3rd output
// of the butterfly maps to the 3 -> 011 -> 110 -> 6th output of the array; and
// so forth.
//
// The algorithm below comes from:
//
//  Dietz, H. (2002) The Aggregate Magic Algorithms. University of Kentucky.
//   http://aggregate.org/MAGIC/#Bit%20Reversal
//
// which decomposes the (m m-1 ... 1) permutation into log m cycles: first swap
// all adjacent bits, then swap every two bits, and so forth.
// 
// Note, however, Dietz' particular algorithm is hard-wired to m=32-bits only:
// so in the example above, 3 is not mapped to 110 but to 1100 0000 0000 0000 
// 0000 0000 0000 0000; so, at the end, I just right-shift the result by 32-m 
// bits to obtain the desired value (in this case, 110 -> 6).
// 
//   x:                      Index to bit-reverse.
//   m:                      How many bits to use when representing the index.
//
//   returns:                Bit-reversed integer value of x, when represented
//                           as a m-bit binary number.
// 
uint32_t bit_reverse(uint32_t x, uint32_t m){
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));

  return ((x >> 16) | (x << 16)) >> (32-m);
}

// This function starts the given tmr_t object.
//
//   timer:                  (pointer to) tmr_t object to start.
//
//  (From: http://stackoverflow.com/questions/7876624/timing-cuda-operations)
//
void timer_start(tmr_t* tm){
  cudaError_t ret;                          // return value of CUDA calls

  if(cuda_try(ret = cudaEventCreate(&tm->start))  ||
     cuda_try(ret = cudaEventCreate(&tm->stop))   ||
     cuda_try(ret = cudaEventRecord( tm->start, 0))){
    stderr_out("cudaEventCreate|Record error: %s\n", cudaGetErrorString(ret));
  }
}

// This function stops the given tmr_t object, annd calculates the elapsed
// time since the call last to start_timer.
//
//   timer:                  (pointer to) tmr_t object to stop.
//
//  (From: http://stackoverflow.com/questions/7876624/timing-cuda-operations)
//
void timer_stop(tmr_t* tm){
  cudaError_t ret;                          // return value of CUDA calls

  if(cuda_try(ret = cudaEventRecord     ( tm->stop, 0))  ||
     cuda_try(ret = cudaEventSynchronize( tm->stop))     ||
     cuda_try(ret = cudaEventElapsedTime(&tm->elapsed, tm->start, tm->stop))){
    stderr_out("cudaEventRecord|Synchronize|ElapsedTime error: %s\n", 
      cudaGetErrorString(ret));
  }
}

// This function returns the number of concurrent blocks in the GPU.
//
//   returns:                Number of concurrent blocks in the GPU (if any)
//
uint32_t how_many_concurrent_blocks(){
  cudaError_t ret;                     // return value of CUDA calls
  int dev, devs;                       // number of devices
  cudaDeviceProp pr;                   // device properties
  uint32_t p;                          // number of concurrent blocks

  // Count how many devices are there. If err or no devices, set p=0 and print.
  if(cuda_try(ret = cudaGetDeviceCount(&devs)) || devs == 0){
    stderr_out("cudaGetDeviceCount error: %s\n", cudaGetErrorString(ret));
    return 0;
  }

  // Get the device properties of the last device
  dev = devs - 1;
  cudaSetDevice(dev);
  cudaGetDeviceProperties(&pr, dev);

  // Compute the number of concurrent blocks according to the formula 
  //
  //   # of SMs x # of warp schedulers per SM
  //
  p = pr.multiProcessorCount * how_many_warp_schedulers(pr.major,pr.minor);

  return p;
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
int how_many_warp_schedulers(int major, int minor){
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
