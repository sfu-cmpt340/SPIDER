import numpy as np;
import pandas as pd;
import os;
import matplotlib.pyplot as plt;
import wfdb;

import simple_denoise;
from simple_denoise import get_valid_segments;

directory_name = 'processed_dat'

#make the directory name if they don't exist

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
    print(f"Directory '{directory_name}' created successfully.")
else:
    print(f"Directory '{directory_name}' already exists.")

#get the file names
DIR = 'CTU-CHB/dat/'
recnos = np.loadtxt(DIR+'RECORDS')
recnos = [int(x) for x in recnos]

for ind in recnos:
    FILEPATH = DIR+str(ind)
    
    #print(FILEPATH)

    signals, fields = wfdb.rdsamp(FILEPATH)

    # transpose the matrix as originally it was vertical
    signals = np.transpose(signals)

    #print(signals[0])

    fetal_hr = signals[0]

    # the total seconds of the file is the length of the file divided by 4 as it was sampled at 4Hz
    ts = np.arange(len(fetal_hr))/4.0

    #get the valid segments (ie the processed segments without the long gaps)
    selected_segments = get_valid_segments(fetal_hr, ts, FILEPATH, verbose=False,
                                               #max_change=15, verbose_details=True
                                              )
    len_s = len(selected_segments)
    new_signal = []

    #add the segments together for the new processed signal
    for i in range(len_s):
        new_signal.extend(selected_segments[i]['seg_hr'])

    #function to write the data
    def write_array_to_dat_file(data_array, file_path):
        with open(file_path, 'w') as file:
            for number in data_array:
                file.write(str(number) + '\n')
    
                
    #write the data to the new directory
    data = new_signal
    file_path = 'processed_dat/'+str(ind)+'.dat'
    write_array_to_dat_file(data, file_path)
    #print(file_path)