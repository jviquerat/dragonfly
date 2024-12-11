#!/bin/bash
#
#SBATCH --job-name=dgf_benchmark
#SBATCH --output=dgf_benchmark.txt
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-node 64
#SBATCH --ntasks-per-core 1
#SBATCH --time=7-00:00:00
#
folder=$1
n_cpu=${2-1}
module load cimlibxx/drl/python3.11
module load openmpi/4.1.1
source ~/scratch/dragonfly/venv/bin/activate
./bench_folder.sh $folder $n_cpu
