#!/bin/bash
python maxcut.py -s square -i step_fn 
python maxcut.py -s square -i power_decay_fn 
python maxcut.py -s triangular -i step_fn 
python maxcut.py -s triangular -i power_decay_fn 