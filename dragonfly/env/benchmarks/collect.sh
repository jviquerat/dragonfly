#!/usr/bin/env bash

#######################################################
# Collect avg.dat results in all folders by renaming them
# with the name of their parent folder, then prune samples
# until correct size for plotting is reached
#######################################################

# Retrieve folder name
input=$1

# Remove possible trailing slash
folder=${input%/}

# List all directories
lst=($(find $folder -type d))

# Create new dir
new_dir="bench_data"
if [ ! -d $new_dir ]; then
    mkdir $new_dir
fi

# Print files
echo "Collecting files:"
echo " "
for f in ${lst[@]}; do

    # Check if folder has avg.dat file in it
    fdat="${f}/avg.dat"
    if [ -f $fdat ]; then
	#echo $fdat

	# Keep after last slash
	f=${f##*/}
	echo $f

	# Remove capital letters
	f=$(echo $f | tr '[:upper:]' '[:lower:]')

	# Copy and rename avg.dat file
	cp $fdat $new_dir/$f

	# Get file size in kb
	dataf=$new_dir/$f
	fsize=`du -k "$dataf" | cut -f1`

	# Make file smaller than 200k
	size=200
	while [[ $fsize -gt $size ]]; do
	    sed -i '0~2d' $dataf
	    fsize=`du -k "$dataf" | cut -f1`
	done
    fi
done
