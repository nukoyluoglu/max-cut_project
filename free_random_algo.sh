#!/bin/bash
for system_size in {0..25}
do
    python maxcut_algo.py -s free -i random -n $system_size
done