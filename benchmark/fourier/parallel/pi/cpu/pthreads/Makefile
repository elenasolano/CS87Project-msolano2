#

# CS87 - Final Project
# Maria-Elena Solano
#
# Makefile for benchmark implementation of pi-DFT on multicore CPU (pthreads).
# Includes the usual 'all' and 'clean' targets, plus a few extra ones for added
# convenience:
#
#   make recompile;                              # Cleans up and recompiles
#   make run-experiments-and-analyze-results;    # Runs the experiments and
#                                                # analyzes the results.
#   make replicate;                              # Runs all of the above
#

.PHONY: all clean recompile run-experiments-and-analyze-results replicate

SHELL    = /bin/bash
CC       = gcc
INCLUDES = 
CFLAGS   = -Wall -g
LDLIBS   = -lm -lpthread
LDFLAGS  = -fopenmp -pthread
TARGET   = fourier-parallel-pi-cpu-pthreads


all: $(TARGET)

$(TARGET): $(TARGET).c
	@$(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@;
clean:
	@$(RM) -rf $(TARGET) $(TARGET).dSYM;


recompile:
	@echo "Recompiling benchmark implementation of pi-DFT on multicore CPU...";
	@make -s clean;
	@make -s;

run-experiments-and-analyze-results:
	@qsub run-experiments-and-analyze-results -v DIR=$(shell pwd);

replicate:
	@echo "";
	@make -s recompile;
	@make -s run-experiments-and-analyze-results;
