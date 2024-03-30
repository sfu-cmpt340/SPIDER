import os
import pandas as pd

dat_path = 'dat'
folder_path = 'processed_dat'
all_data = []

def map_score(score, key):
    thresholds = {
        'Apgar1': [(6, 0), (8, 1)],
        'Apgar5': [(8, 0), (9, 1)]
    }
    for threshold, mapped_value in thresholds.get(key, [(float('inf'), 2)]):
        if score <= threshold:
            return mapped_value
    return 2

remap = False

for file_name in os.listdir(dat_path):
    if file_name.endswith('.hea'):
        dat_file_path = os.path.join(folder_path, os.path.splitext(file_name)[0] + '.dat')
        if (not os.path.exists(dat_file_path)) or os.path.getsize(dat_file_path) <= 0:
            print(f"Empty dat at {dat_file_path}")
            continue
        data = {}
        with open(os.path.join(dat_path, file_name), 'r') as file:
            outcome_measures_started = False
            for line in file:
                line = line.strip()
                if line.startswith('#-- Outcome measures'):
                    outcome_measures_started = True
                elif line.startswith('#') and outcome_measures_started:
                    parts = line.split()
                    if len(parts) == 2:
                        key = parts[0][1:]
                        value = parts[1]
                        if key in ['Apgar1', 'Apgar5'] and remap:
                            data[key + 'unmapped'] = int(value)
                            value = map_score(int(value), key)
                        data[key] = value
                    else:
                        outcome_measures_started = False
                else:
                    if outcome_measures_started:
                        break
        data['filename'] = file_name.replace(".hea", ".png")
        all_data.append(data)

df = pd.DataFrame(all_data)
print(df)
# print(df['Apgar1'].value_counts())
# print(df['Apgar5'].value_counts())
df.to_csv('outcomes.csv', index=False)
