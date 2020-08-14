#!/bin/bash

for i in {2..5}; do
    python3 quantum_algorithms.py -s square -f 1.0 -i step_fn -n $i $i &
    # python3 quantum_algorithms_MT.py -s square -f 1.0 -i step_fn -n $i $i &
done
wait