#

# CS87 - Final Project
# Maria-Elena Solano
#
# Makefile for benchmark implementation of pi-DFT on the Xeon Phi using OpenMP.
# Includes the usual 'all' and 'clean' targets, plus a few extra ones for added
# convenience:
#
#   make recompile;                              # Cleans up and recompiles
#   make run-experiments-and-analyze-results;    # Runs the experiments and
#                                                # analyzes the results.
#   make replicate;                              # Runs all of the above
#
#
# Note: compilation requires the Intel C compiler, and experiments will only
#       run if there is a Xeon Phi available in the machine.
#

.PHONY: all clean recompile run-experiments-and-analyze-results replicate

SHELL    = /bin/bash
CC       = icc
INCLUDES = 
CFLAGS   = -Wall -g
LDLIBS   = -lm
LDFLAGS  = -openmp -mmic
TARGET   = fourier-parallel-pi-xeonphi-openmp


all: $(TARGET)

$(TARGET): $(TARGET).c
	@if ! type icc >/dev/null 2>&1; then \
	  echo -e "\x1B[1;31mError:\x1B[0m no Intel C compiler available!" \
	          "Cannot compile for the Xeon Phi here."; \
	else \
	  $(CC) $(CFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@; \
	fi;
clean:
	@$(RM) -rf $(TARGET) $(TARGET).dSYM;


recompile:
	@echo "Recompiling benchmark implementation of pi-DFT on Xeon Phi...";
	@make -s clean;
	@make -s;

run-experiments-and-analyze-results:
	@sbatch run-experiments-and-analyze-results;

replicate:
	@echo "";
	@make -s recompile;
	@make -s run-experiments-and-analyze-results;
