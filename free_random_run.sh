#!/bin/bash

for system_size in {4..25}
do
    python maxcut_exact.py -s free -i random -n $system_size
    python maxcut_algo.py -s free -i random -n $system_size
    python maxcut_plot_system.py -s free -i random -n $system_size
done
python maxcut_plot_summary.py -s free -i random