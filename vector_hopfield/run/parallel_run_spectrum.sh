#!/bin/bash

N=${1?system_size:}
d=${2?dimension_of_spins:}
alpha=${3?capacity:}
nsample_per_batch=${4?nsample_per_batch:}
option_initialization=${5?option_init:}
nbatches=${6?nbatches:}

for ((i=0; i<nbatches; i++)); do

let sample_index_start=i*nsample_per_batch

echo $N $d $alpha $option_initialization $sample_index_start $nsample_per_batch > input_for_run_spectrum.txt

python3 run_spectrum.py &

sleep 1

done


