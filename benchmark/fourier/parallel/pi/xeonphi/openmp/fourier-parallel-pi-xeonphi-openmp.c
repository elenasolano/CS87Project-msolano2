// CS 87 - Final Project
// Maria-Elena Solano
//
// Radix-2 Cooley-Tukey Fourier Transform on C^n - parallel 'pi' version
// on Xeon Phi
//

#include <stdio.h>                     // C's standard I/O library
#include <stdlib.h>                    // C's standard library
#include <stdint.h>                    // C's exact width int types
#include <string.h>                    // C's standard string library
#include <time.h>                      // C's time types (for random init)
#include <math.h>                      // C's math library
#include <unistd.h>                    // C's POSIX API
#include <sys/time.h>                  // System time types
#include <omp.h>                       // OpenMP library


// macro/constant definitions
#define perror_out(X)                  perror(X), fflush(stderr)
#define stderr_out(...)                fprintf(stderr, __VA_ARGS__), \
                                        fflush(stderr)
#define print_out(...)                 printf(__VA_ARGS__), fflush(stdout)
#define copy_to(X,Y,Z)                 memcpy((X), (Y), sizeof(data_t)*(Z))
                                        

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
  data_t*   in;                        // input data
  data_t*   out;                       // output
  data_t*   tmp_in;                    // input scratchpad
  data_t*   tmp_out;                   // output scratchpads
  uint32_t  test_mode;                 // test mode?
  uint32_t  no_header;                 // no timing headers?
} tr_t;

// simple encapsulating struct for timing
typedef struct timer{
  double  start, stop;                 // start and stop time
  double  elapsed;                     // elapsed time (if any), in millisecs
} tmr_t;

// function declarations
// setting up the transform
int  setup_from_args (tr_t* t, int argc, char** argv);
void show_usage      ();

// running the transform
int  run             (tr_t  t);
int  initialize_data (tr_t* t);
void cleanup_data    (tr_t* t);
void butterfly       (data_t* out, data_t* in, uint32_t size, uint32_t N);
void butterfly_left  (data_t* out, data_t* in, uint32_t size, uint32_t N);
void butterfly_right (data_t* out, data_t* in, uint32_t size, uint32_t N);

// complex arithmetic
data_t add  (data_t a, data_t b);
data_t sub  (data_t a, data_t b);
data_t mul  (data_t a, data_t b);
data_t omega(uint32_t N, uint32_t k);

// printing and verifying
void print_input     (tr_t* t);
void print_output    (tr_t* t);
void verify_results  (tr_t* t);

// timing
void timer_start(tmr_t* tm);
void timer_stop (tmr_t* tm);

// misc
void     swap_scratchpads(tr_t* t);
int      is_power_of_two (int x);
uint32_t hash            (uint32_t x);
uint32_t ilog2           (uint32_t x);
uint32_t bit_reverse     (uint32_t x, uint32_t N);


int main(int argc, char** argv){
  tr_t t;                              // main transform object

  // (1) setup the transform from the command line args (or ret -1 if err)
  if(setup_from_args(&t, argc, argv)){
    exit(EXIT_FAILURE);
  }

  // (2) run the transform according to the given args (or ret -1 if err)
  if(run(t)){
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
      // If option -t, set t.N to 8 and t.test to 'true', and notify user.
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
        break;
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
  // If too many processors, return -1.
  if(t->P > 61){
    stderr_out("Too many processors! (Only up to 61 cores available)\n");
    show_usage();
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

  // We need:
  // - N entries for the input, and
  if((t->in = (data_t*)malloc(sizeof(data_t)*t->N)) == NULL){
    perror_out("malloc error");
    goto err;
  }
  // - N entries for the output.
  if((t->out = (data_t*)malloc(sizeof(data_t)*t->N)) == NULL){
    perror_out("malloc error");
    goto err;
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

  // Free all the allocated data (if any)
  if(t->in  != NULL){ free(t->in);   t->in  = NULL; }
  if(t->out != NULL){ free(t->out);  t->out = NULL; }
  
  return;
}

// This function shows the cmdline interface usage.
// 
void show_usage(){
  print_out("\nusage:\n"
            "  fourier-parallel-pi-xeonphi-openmp { -n <n> -p <p> | -t }\n"
            "\noptions:\n"
            "  -n <n>     power of two input size.\n"
            "  -p <p>     power of two number of processors (less than n).\n"
            "  -o         omit timing headers\n"
            "  -t         compare against precomputed input/output.\n"
            "\n");
}


// This function runs the given transform on P separate processors. Returns 0
// if success or -1 if err.
//
//   t:                      copy of tr_t object to use
//
//   returns:                0 if success, or -1 if error.
// 
int run(tr_t t_shared){
  uint32_t size, iter, offset, bit;         // butterfly size, offset and iter,
  uint32_t which_butterfly, which_half;     // which butterfly, and which half,
  tmr_t    tm_funnel, tm_tube;              // tmr_t objects (for timing)
  tr_t     t;                               // per-thread copy of shared tr_t.

  
  // Initialize the data (or ret -1 err, cleaning up any malloc'd data first)
  if(initialize_data(&t_shared)){
    cleanup_data(&t_shared);
    goto err;
  }

  // Spawn P threads
  omp_set_num_threads(t_shared.P);
  #pragma omp parallel private(size, iter, offset, bit, \
                               which_butterfly, which_half, \
                               tm_funnel, tm_tube, t)
  {
    
    // Populate the local tr_t object, and determine which Pi is this.
    t    = t_shared;
    t.Pi = omp_get_thread_num();

    // We need:
    // - N entries for the input scratchpad, and
    if((t.tmp_in = (data_t*)malloc(sizeof(data_t)*t.N)) == NULL){
      perror_out("malloc error");
    }
    // - N entries for the output scratchpad.
    if((t.tmp_out = (data_t*)malloc(sizeof(data_t)*t.N)) == NULL){
      perror_out("malloc error");
    }

    // Copy the full input to the input scratchpad.
    copy_to(t.tmp_in, t.in, t.N);



    // (1) Tree stage: 
    // ---------------

    // Start the timer for the funnel stage
    timer_start(&tm_funnel);

    // For the first log P iters
    // (i.e. butterfly sizes N,     N/2,       ..., 2(N/P),
    //       iteration stage log P, log P - 1, ..., 1),
    for(size = t.N, iter = ilog2(t.P); size > t.N/t.P; size /= 2, iter--){

      // Determine _which_ butterfly to compute, depending on (Pi >> iter)
      which_butterfly = (t.Pi >> iter);

      // Determine _which_ segment_ of size 'size' to compute the butterfly on
      offset = which_butterfly * size;

      // Determine _which half_, depending on the parity of the 1st, 2nd, ...
      // most significant bit of Pi.
      which_half = ((t.Pi >> (iter - 1)) % 2 == 0);

      // And compute the butterfly accordingly
      if(which_half){
        butterfly_left(
          t.tmp_out + offset,  // output to the _1st_ half of the scratchpad
          t.tmp_in  + offset,  // and reading from the input scratchpad.
          size, t.N);
      }
      else{
        butterfly_right(
          t.tmp_out + offset,  // output to the _2nd_ half of the scratchpad
          t.tmp_in  + offset,  // and reading from the input scratchpad.
          size, t.N);      
      }

      // And swap scratchpads, so that the new results become the input for
      // the next iteration.
      swap_scratchpads(&t);
    }

    // Stop the timer for the tree stage
    timer_stop(&tm_funnel);


    // (2) Cylinder stage
    // ------------------

    // Start the timer for the tube stage
    timer_start(&tm_tube);

    // For the last log N/P iters
    // (i.e. butterfly sizes N/P, (N/P)/2, ..., 2), 
    // we work only on the portion of the output assigned to this processor
    for(size = t.N/t.P; size > 1; size /= 2){

      // Compute 1, 2, ..., (N/P)/2 butterflies of size N/P, (N/P)/2, ..., 2 
      // over _consecutive_ intervals across the assigned segment (of size N/P)
      // (which recall starts from (t.N/t.P)).
      for(offset=(t.N/t.P)*t.Pi; offset<(t.N/t.P)*t.Pi+t.N/t.P; offset+=size){
        butterfly(
          t.tmp_out + offset,       // output to the _assigned_ segment,
          t.tmp_in  + offset,       // and reading from the _assigned_ segment.
          size, t.N);
      }

      // And swap scratchpads, so that the new results become the input for
      // the next iteration.
      swap_scratchpads(&t);
    }

    // Stop the timer for the tube stage
    timer_stop(&tm_tube);


    // If not in test mode, show elapsed times (total, funnel and tube stages)
    if(!t.test_mode && t.Pi == 0){
      if(!t.no_header){ 
        print_out("n\tp\ttime (total)\t"
                  "time (stage 1)\ttime (stage 2)\n"); };
      print_out("%u\t%u\t%lf\t%lf\t%lf\n", 
                t.N, t.P, tm_funnel.elapsed + tm_tube.elapsed, 
                tm_funnel.elapsed, tm_tube.elapsed);
    }
    // Otherwise, copy the result from the _assigned_ segment of the input
    // scratchpad (which will contain the new results at this point) to the 
    // output, in bit-reversed order.
    else{
      for(bit = (t.N/t.P)*t.Pi; bit < (t.N/t.P)*t.Pi+t.N/t.P; bit++)
        t.out[bit_reverse(bit, ilog2(t.N))] = t.tmp_in[bit];
    }
    

    // Free the data
    if(t.tmp_in  != NULL){ free(t.tmp_in);   t.tmp_in  = NULL; }
    if(t.tmp_out != NULL){ free(t.tmp_out);  t.tmp_out = NULL; }

  }

  // If in test mode, print out the result, and verify that it is correct.
  if(t_shared.test_mode){
    print_output(&t_shared);
    verify_results(&t_shared);
  }

  // Cleanup the data, and return
  cleanup_data(&t_shared);
  return 0;

err:
  stderr_out("Could not run the transform\n");
  return -1;
}


// This function computes an n-point butterfly over the given input array,
// placing the results on the given output array.
//
//   out,in:                 output and input arrays
//   size:                   size of the butterfly
//   N:                      input size of the broader FFT
//
void butterfly(data_t* out, data_t* in, uint32_t size, uint32_t N){

  // Compute the left butterfly (note it only writes on the _1st_ half of out)
  butterfly_left (out, in, size, N);

  // And the left butterfly (note it only writes on the _2nd_ half of out)
  butterfly_right(out, in, size, N);

  return;
}

// This function computes the left half of an n-point butterfly. Note it only
// writes to the _first_ half of the output.
//
//   out,in:                 output and input arrays
//   size:                   size of the butterfly
//   N:                      input size of the broader FFT
//
void butterfly_left(data_t* out, data_t* in, uint32_t size, uint32_t N){
  uint32_t bit;
      
  // For each bit in the segment
  for(bit = 0; bit < size/2; bit++){
    out[bit] =                              //  set the current (output) bit to
      add(                                  //   the sum of
        in[bit],                            //    the current (input) bit, and
        in[bit+size/2]);                    //    the (size/2)-next one.
  }

  return;
}

// This function computes the right half of an n-point butterfly. Note it only
// writes to the _second_ half of the output.
//
//   out,in:                 output and input arrays
//   size:                   size of the butterfly
//   N:                      input size of the broader FFT
//
void butterfly_right(data_t* out, data_t* in, uint32_t size, uint32_t N){
  uint32_t bit;
      
  // For each bit in the segment (note we are offsetting out by size/2!
  for(out += size/2, bit = 0; bit < size/2; bit++){
    out[bit] =                              //  set the current (new) bit to
      mul(                                  //   the multiplication of
        sub(                                //    the sum of the
          in[bit],                          //     the (size/2)-prev bit, and
          in[bit+size/2]),                  //     the next one
        omega(N,                            //    times the Nth root of unity,
              bit*(N/size)));               //     to the power of 0,1,2 first,
  }                                         //     0,2,4 second, etc.

  return;
}

// This function returns (by value) the sum of the two given complex numbers.
//
//   a,b:                    operands (x+yi,z+wi) (of type data_t)
//
//   returns:                the sum a+b = (x+z)+(y+w)i
// 
data_t add(data_t a, data_t b){
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
data_t sub(data_t a, data_t b){
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
data_t mul(data_t a, data_t b){
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
data_t omega(uint32_t N, uint32_t k){
  data_t o;

  o.re =  cos(2.0*M_PI/N*k);
  o.im = -sin(2.0*M_PI/N*k);

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


// This function starts the given tmr_t object.
//
// Note: using OpenMP's omp_get_wtime for portability.
//
//   timer:                  (pointer to) tmr_t object to start.
//
void timer_start(tmr_t* tm){
  tm->start = omp_get_wtime();

  return;
}

// This function stops the given tmr_t object, annd calculates the elapsed
// time since the call last to start_timer, in milliseconds.
//
// Note: using OpenMP's omp_get_wtime for portability.
//
//   timer:                  (pointer to) tmr_t object to stop.
//
void timer_stop(tmr_t* tm){
  tm->stop    = omp_get_wtime();
  tm->elapsed = (tm->stop - tm->start)*1000.0;

  return;
}


// This function swaps the input and output scratchpads of the given tr_t obj.
//
//   t:                      (ptr to) tr_t object to use.
//
void swap_scratchpads(tr_t* t){
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
uint32_t ilog2(uint32_t x){  
  static const int lookup_table[32] = {
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
