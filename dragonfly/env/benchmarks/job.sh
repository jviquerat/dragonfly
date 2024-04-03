#!/bin/bash
#
#SBATCH --job-name=dgf_benchmark
#SBATCH --output=dgf_benchmark.txt
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=4-00:00:00
#
folder=$1
source ~/scratch/dragonfly/venv/bin/activate
./bench_folder.sh $folder
