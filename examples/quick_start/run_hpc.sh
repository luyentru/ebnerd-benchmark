#!/bin/sh
### General options
### -- set the job Name --
#BSUB -J paula_150GB
### -- ask for number of cores (default: 1) --
#BSUB -q gpua100
#BSUB -n 1
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=150GB]"

source venv/bin/activate
cd examples/00_quick_start

module load cuda/12.2
echo $CUDA_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

python3 our_nrms.py