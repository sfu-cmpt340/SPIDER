import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Based on: https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def rec_plot(data, eps=0.1, steps=10):
    d = pdist(data[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z


input_directory = 'processed_dat'
output_directory = 'recurrence_plots'

file_names = os.listdir(input_directory)

for file_name in file_names:
    file_path = os.path.join(input_directory, file_name)
    if os.path.getsize(file_path) > 0:  # Check if file is not empty
        data = np.loadtxt(file_path)
        rp = rec_plot(data)

        fig = plt.figure(figsize=(8, 6))
        plt.imshow(rp, cmap='binary', origin='lower')
        plt.gca().set_facecolor('white')
        plt.tight_layout()
        plot_file_name = os.path.splitext(file_name)[0] + '_recurrence_plot.png'
        plot_path = os.path.join(output_directory, plot_file_name)
        plt.savefig(plot_path, dpi=75)  
        plt.close()
    else:
        print('File is empty:', file_path)