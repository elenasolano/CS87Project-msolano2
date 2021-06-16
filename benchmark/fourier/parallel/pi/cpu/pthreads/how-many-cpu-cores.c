// CS 87 - Final Project
// Maria-Elena Solano
//
// This utility simply counts how many CPU cores are in this node
//

#include <stdio.h>                     // C's standard I/O library
#include <stdlib.h>                    // C's standard library
#include <unistd.h>                    // C's POSIX API


// macro/constant definitions
#define perror_out(X)                  perror(X), fflush(stderr)
#define stderr_out(...)                fprintf(stderr, __VA_ARGS__), \
                                        fflush(stderr)
#define print_out(...)                 printf(__VA_ARGS__), fflush(stdout)


int main(int argc, char** argv){
  long int p;                          // number of cores

  // (1) try retrieve the number of cores (or ret -1 if err)
  if((p = sysconf(_SC_NPROCESSORS_ONLN)) == -1){
    exit(EXIT_FAILURE);
  }

  // (2) print it out
  print_out("%ld\n", p);

  // (3) and return
  exit(EXIT_SUCCESS);
}
