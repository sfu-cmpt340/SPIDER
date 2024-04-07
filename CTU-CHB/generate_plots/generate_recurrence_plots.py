import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Based on: https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def rec_plot(data, eps=0.3, steps=15):
    d = pdist(data[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


input_directory = '../processed_dat'
output_directory = '../dat_recurrence_plots'

file_names = os.listdir(input_directory)

for i in range(0, 545): # 545 files in the directory
    file_name = file_names[i]
    file_path = os.path.join(input_directory, file_name)
    if os.path.getsize(file_path) > 0:  # Check if file is not empty
        data = np.loadtxt(file_path)
        rp = rec_plot(data)

        fig = plt.figure(figsize=(2, 2))
        plt.imshow(rp, cmap='binary')
        plt.axis('off')
        plt.tight_layout()
        plot_file_name = os.path.splitext(file_name)[0] + '.png'
        plot_path = os.path.join(output_directory, plot_file_name)
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(plot_path, dpi=75)  
        plt.close()
    else:
        print('File is empty:', file_path)