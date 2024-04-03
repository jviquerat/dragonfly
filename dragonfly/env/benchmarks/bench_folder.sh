#!/usr/bin/env bash

#######################################################
# Run all .json scripts contained in a provided folder recursively
# You need to source virtual environment before starting
#######################################################

# Retrieve folder name
input=$1

# Remove possible trailing slash
folder=${input%/}

# List all json files
lst=($(find $folder -type f -name '*.json'))

# Print files
echo "Benchmarking files:"
echo " "
for f in ${lst[@]}; do
    echo $f

    # Remove possible leading characters
    name=${f#..}
    name=${name#/}

    # Remove json extension
    name=${name%.*}

    # Convert slashes to underscore
    name=${name//\//_}

    # Run with output in dedicated file
    dgf --train $f &> $name &

    # Retrieve background pid
    pid = $!
done

wait $pid
wait
