#

# CS87 - Final Project
# Maria-Elena Solano
#
# Makefile for benchmark implementation of pi-DFT on a GPU using CUDA.
# Includes the usual 'all' and 'clean' targets, plus a few extra ones for added
# convenience:
#
#   make recompile;               # Cleans up and recompiles
#   make run-experiments;         # Runs the experiments
#   make analyze-results;         # Analyzes the results
#   make replicate;               # Runs all of the above
#

.PHONY: all clean recompile run-experiments analyze-results replicate

SHELL        = /bin/bash
NVCC         = nvcc -ccbin=/usr/bin/g++
HOSTCFLAGS   = --compiler-options -g,-Wall
DEVICECFLAGS = --ptxas-options    -g,-O0,-dlcm=cg
TARGET       = fourier-parallel-pi-gpu-cuda


all: $(TARGET)

$(TARGET): $(TARGET).cu
	@if ! type nvcc >/dev/null 2>&1; then \
	  echo -e "\x1B[1;31mError:\x1B[0m no NVIDIA C compiler available!" \
	          "Cannot compile for the GPU here."; \
	else \
	  $(NVCC) $^ -o $@ $(HOSTCFLAGS) $(DEVICECFLAGS) >/dev/null 2>&1; \
	fi;
clean:
	@$(RM) -rf $(TARGET) $(TARGET).dSYM;


recompile:
	@echo "Recompiling benchmark implementation of pi-DFT on the GPU...";
	@make -s clean;
	@make -s;

run-experiments:
	@./run-experiments;

analyze-results:
	@./analyze-results;

replicate:
	@echo "";
	@make -s recompile;
	@make -s run-experiments;
	@make -s analyze-results;
