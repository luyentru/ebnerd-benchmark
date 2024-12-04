#!/bin/sh
### General options
#BSUB -q gpua100                          # Use the A100 queue
#BSUB -J llama_embeddings                 # Job name
#BSUB -n 4                                # Number of CPU cores
#BSUB -gpu "num=1:mode=exclusive_process" # Request 1 GPU
#BSUB -R "rusage[mem=50GB]"               # Request 50GB system memory
#BSUB -W 24:00                            # Maximum walltime: 24 hours
#BSUB -o base_implementation_%J.out       # Output file
#BSUB -e base_implementation_%J.err       # Error file
#BSUB -B                                  # Send notification at job start
#BSUB -N                                  # Send notification at job completion

# Load CUDA 12.2 and set XLA_FLAGS
git checkout llama-embeddings
module load cuda/12.2
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Activate your virtual environment
source llam-venv/bin/activate

# Run your implementation
python examples/quick_start/nrms_ebnerd.py