#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sort and clip the outcome from the given filename
"""

import sys
import os
import numpy as np

assert len(sys.argv) = 2 
filename = sys.argv[1]
fr = open(filename, 'r').readlines()
lines = fr.readlines()

def sorted_by_num(line):
    return int(line.split(',')[0])

sorted_line = sorted(lines, key=sorted_by_num)
for line in sorted_line:
    splitted_line = line.split(',')
    value_string = ','.join([str(splitted_line[0]), str(np.clip(float(splitted_line[1]), 0.005, 0.995))])
    with open('clipped_outcome', 'a') as f:
        f.write(value_string+'\n')
