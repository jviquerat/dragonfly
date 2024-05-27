#!/bin/bash
#
#SBATCH --job-name=dgf_benchmark
#SBATCH --output=dgf_benchmark.txt
#SBATCH --partition=main
#
#SBATCH --nodes 1
#SBATCH --ntasks 12
#SBATCH --ntasks-per-node 12
#SBATCH --ntasks-per-core 1
#SBATCH --time=7-00:00:00
#
folder=$1
n_cpu=${2-1}
source ~/dragonfly/venv/bin/activate
./bench_folder.sh $folder $n_cpu
