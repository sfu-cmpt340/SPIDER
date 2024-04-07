import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Baseline FHR value
# Measured as the mean of all signal values
def getBaseline(fhr_signal):
    baseline = np.mean(fhr_signal)
    # print(baseline)
    
    return baseline

# Plot the singal
def printWaveform(fhr_signal, sampling_rate, baseline, time):
    time = np.arange(len(fhr_signal)) / sampling_rate
    plt.figure(figsize=(12, 2))
    plt.plot(time, fhr_signal, color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('FHR (bpm)')
    plt.title('Fetal Heart Rate Signal')
    plt.grid(True)
    plt.axhline(y=baseline, color='red', label='Baseline')
    plt.ylim(50, 200)
    plt.xlim(time[0], time[-1])
    plt.show()

# Separate the FHR singal into segments
# Based on the intersections where the FHR meets the Baseline
# Get all segments where the FHR is above the baseline
# Get all segments where the FHR is below the baseline
def getSegments(fhr_signal, baseline):
    current_segment = []
    segments_above_baseline = []
    segments_below_baseline = []
    # Check if first signal is above the baseline
    above_baseline = fhr_signal[0] > baseline

    # Loop through entire FHR signal
    for signal in fhr_signal:
        # Check if current signal is above the baseline
        is_above_baseline = signal > baseline
        # Check if the signal has changed sides of the baseline
        # I.e an intersection has been reached
        if is_above_baseline != above_baseline:
            # If the signal changed and the signal is above the baseline
            # Add segment to segments_above_baseline
            # Otherwise add segment to segments_below_baseline
            if above_baseline:
                segments_above_baseline.append(current_segment)
            else:
                segments_below_baseline.append(current_segment)
            
            current_segment = [signal]
            above_baseline = is_above_baseline
        else:
            # Add signal to current segment
            current_segment.append(signal)

    # Add the last segment to its corresponding list
    if above_baseline:
        segments_above_baseline.append(current_segment)
    else:
        segments_below_baseline.append(current_segment)
        
    return segments_above_baseline, segments_below_baseline

# FHR accelerations
# Defined as an increase in FHR signal between two intersections on the baseline
# such that the highest point in the segment is at least 15 b.p.m above the baseline and the segment is 15 (seconds)
def getAccelerations(segments_above_baseline, window_size, threshold_bpm, baseline, time):
    num_accelerations = 0

    # Iterate through all segments above the baseline
    for segment in segments_above_baseline:
        max_signal = np.max(segment)
        # Check if the segment length is at least 15 and the max signal in the segment is 15 b.p.m above the baseline
        if len(segment) / 4 >= window_size and max_signal >= baseline + threshold_bpm:
            num_accelerations += 1

    AC = num_accelerations / len(time)
    # print(AC)
    
    return AC

# FHR decelerations
# Defined as an decrease in FHR signal between two intersections on the baseline
# such that the lowest point in the segment is at least 15 b.p.m below the baseline and the segment is 15 (seconds)
# If the segment is at lesat 120 seconds / 2 minutes, count as prolongued deceleration
def getDecelerations(segments_below_baseline, window_size, prolongued_window_size, threshold_bpm, baseline, time):
    num_decelerations = 0
    num_prolonged_decelerations = 0
    
    # Iterate through all segments below the baseline
    for segment in segments_below_baseline:
        min_signal = np.min(segment)

        # Check if the segment length is at least 15 and the min signal in the segment is 15 b.p.m below the baseline
        if len(segment) / 4 >= window_size and min_signal <= baseline - threshold_bpm:
            num_decelerations += 1
            # Check if segment length is at least 120
            if len(segment) / 4 >= prolongued_window_size:
                num_prolonged_decelerations += 1

    DC = num_decelerations / len(time)
    DP = num_prolonged_decelerations / len(time)
    # print(DC)
    # print(DP)
    
    return DC, DP

# Variability
# Defined as the difference between the max signal and the min signal within a given time frame
# Short term defined as a 1 minute time frame
# Abnormality defined as the variability being less than 5 and greater than 25
def getShortTermVariability(fhr_signal, sampling_rate, time):
    segment_length = 60 * sampling_rate
    num_abnormal_stv = 0

    stv_values = []
    # Iterate through the FHR signal
    for i in range(0, len(fhr_signal), segment_length):
        # Segment the signal based on the time frame
        segment = fhr_signal[i:i+segment_length]
        max_signal = np.max(segment)
        min_signal = np.min(segment)

        # Add the variability of the time frame to the list
        stv = max_signal - min_signal
        stv_values.append(stv)

        # Check if the variability is abnormal
        if stv < 5 or stv > 25:
            num_abnormal_stv += 1

    MSTV = np.mean(stv_values)
    # print(MSTV)
    ASTV = num_abnormal_stv / len(time)
    # print(ASTV)
    
    return MSTV, ASTV

# Variability
# Defined as the difference between the max signal and the min signal within a given time frame
# Long term defined as a 5 minute time frame
# Abnormality defined as the variability being less than 5 and greater than 25
def getLongTermVariability(fhr_signal, sampling_rate, time):
    segment_length = 300 * sampling_rate
    num_abnormal_ltv = 0

    ltv_values = []
    # Iterate through the FHR signal
    for i in range(0, len(fhr_signal), segment_length):
        # Segment the signal based on the time frame
        segment = fhr_signal[i:i+segment_length]
        max_signal = np.max(segment)
        min_signal = np.min(segment)

        # Add the variability of the time frame to the list
        ltv = max_signal - min_signal
        ltv_values.append(ltv)

        # Check if the variability is abnormal
        if ltv < 5 or ltv > 25:
            num_abnormal_ltv += 1

    MLTV = np.mean(ltv_values)
    # print(MLTV)
    ALTV = num_abnormal_ltv / len(time)
    # print(ALTV)
    return MLTV, ALTV

# Define folder path for CTG data
curr_dir = os.getcwd()
folder_path = os.path.join(curr_dir, 'processed_dat')

# Define variables for CTG data/feature extraction
window_size = 15
prolongued_window_size = 120
threshold_bpm = 15
sampling_rate = 4

# Define column names for DataFrame
columns = ['baseline value', 'accelerations', 'decelerations', 'prolongued_decelerations',
            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability']

data_list = []

# Iterate through all the files in the processed data
for filename in os.listdir(folder_path):
    if filename.endswith(".dat"):
        file_path = os.path.join(folder_path, filename)
    
        with open(file_path, 'rb') as file:
            fhr_signal = np.fromfile(file, dtype=np.float64)
            if len(fhr_signal) > 0:
                LB = getBaseline(fhr_signal)
                time = np.arange(len(fhr_signal)) / sampling_rate

                # printWaveform(fhr_signal, sampling_rate, LB, time)
                
                segments_above_baseline, segments_below_baseline = getSegments(fhr_signal, LB,)
                AC = getAccelerations(segments_above_baseline, window_size, threshold_bpm, LB, time)
                DC, DP = getDecelerations(segments_below_baseline, window_size, prolongued_window_size, threshold_bpm, LB, time)
                
                MSTV, ASTV = getShortTermVariability(fhr_signal, sampling_rate, time)
                MLTV, ALTV = getLongTermVariability(fhr_signal, sampling_rate, time)
                
                data = {'baseline value': LB,
                        'accelerations': AC,
                        'decelerations': DC,
                        'prolongued_decelerations': DP,
                        'abnormal_short_term_variability': ASTV,
                        'mean_value_of_short_term_variability': MSTV,
                        'percentage_of_time_with_abnormal_long_term_variability': ALTV,
                        'mean_value_of_long_term_variability': MLTV}
                
                data_list.append(data)
# Convert list to Pandas DataFrame
df = pd.DataFrame(data_list, columns=columns)
# print(df)

# Convert to csv
df.to_csv('CTU-CHB_data.csv', index=False)

