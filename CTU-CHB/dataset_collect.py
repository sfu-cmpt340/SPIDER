import os
import pandas as pd

folder_path = 'dat'
all_data = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.hea'):
        data = {}
        with open(os.path.join(folder_path, file_name), 'r') as file:
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
df.to_csv('outcomes.csv', index=False)
