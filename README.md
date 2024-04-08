
# Fetal Health Classification from Cardiotocography Data
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/59947126/559234c8-60a4-4a66-862d-5a1c8b7c89cf" width="10%" height="10%">

# Introduction ✏️
Cardiotocography (CTG) is a method commonly used to assess fetal heart rate with ultrasound during labour, and is often used to determine if a fetus is at risk of hypoxia. We evaluated methods on two different datasets to create models that accurately predict the health of a fetus. Our models on the first dataset include testing several classification models to predict/classify the health of the fetus, and identify which model is most effective. On the second dataset, we preprocess biosignals into usable data, then represent as images and train a DCNN to predict fetal health. We also venture into feature extraction to test the efficacy of methods used in the first dataset on the second

## Important Links 🔗

| [Tabulated Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data) | [CTU-CHB dataset](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW38TA3X) | [Project report](https://www.overleaf.com/project/65a57b95a9883102c00a9e4b) | [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EXfKWdGF-QBCtgFGivjFPycBfwZrCIoGwRcEkODk1pRWkw?e=M1zm8O) |

## Video/demo/GIF 📽️
https://drive.google.com/file/d/1HbFV1k-9HB5zfbAsMjozX7fhjIk0FmpX/view

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)


<a name="demo"></a>
## 1. Demo 📝

### Layout Overview

```
fetal-health-classification
├── CTU-CHB                     ## CTU-CHB database
│   ├── dat                         ## Unprocessed raw dat waveforms
│   ├── dat_cwt                     ## cwt plots
│   ├── dat_recurrence_plots        ## recurrence plots
│   ├── dat_spectrogram             ## spectrogram plots
│   ├── DCNN                        ## DCNN Model training and testing
│   ├── Feature Extraction          ## Feature extraction files for tabulation
│   ├── generate_plots              ## Files for generating plots
│   ├── processed_dat               ## Processed and cleaned waveforms
│   ├── outcomes.csv                ## CTU-CHB metadata and labels
│   ├── waveform_processing.py      ## processing waveforms
|
├── TabulatedCTG                ## Tabulated CTG dataset
│   ├── Classify CTU-CHB            ## ML methods on CTU-CHB
│   ├── fetal_health.csv            ## Tabulated Dataset
│   ├── ML Methods.py               ## Machine learning methods
│   ├── NN.py                       ## Neural Network classifier
|
├── requirements.txt            ## Setup requirements
```
### Setup Requirements
Ensure requirements in the Installation section are met.

### 1) Tabulated Dataset Demo

Comparing different machine learning models with various physiology signals to assess the status of fetuses.
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/d7c0f112-893e-4353-8edd-fc59dad33bab" width="50%" height="50%">

To run ML methods on fetal_health.csv: `
python TabulatedCTG/MLMethods.py
`

### 2) CTU-CHB Demo

Analysis of FHR waveforms with the [Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/).

### Pre-processing

- Run `python3 CTU-CHB/waveform_processing.py` on dataset downloaded into `/dat`

Example signal before and after preprocessing:

<img width="1049" alt="Screenshot 2024-04-06 at 1 27 25 PM" src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/29849456/0449a66b-eedd-4462-9eec-8e17bbd00d40">

### Recurrence Plots
- Run the program: `python3 CTU-CHB/generate_plots/generate_recurrence_plots.py`

Here is one sample recurrence plot:

![image](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/f5d9582d-7793-4a3f-bafe-6b67b58a3331)


### DCNN Demo (Pretrained)

To just test the DCNN model, run the test on the prepared test split: `python CTU/CHB/DCNN/python test.py`

![Screenshot 2024-04-08 144435](https://github.com/sfu-cmpt340/fetal-health-classification/assets/59947126/5e4f73de-fcd8-4481-bcaa-b38b95c73a2c)

Instructions for training the model are in the Reproduction section below.

### Feature Extraction

- Feature extraction: `python3 "CTU-CHB/Feature Extraction/Feature_Extraction.py"`
- Test ML methods from tabulated dataset on new dataset: `python3 "TabulatedCTG/Classify_CTU-CHB.py"`

![CTU_ML_Predictions](https://github.com/sfu-cmpt340/fetal-health-classification/assets/59947126/dcae53c0-33a6-4063-a64b-c6ecb760a797)

<a name="installation"></a>
## 2. Installation

Install requirements: `pip install -r requirements.txt`
- DCNN requires tensorflow to be setup.

Download the `fetal-health.csv`: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data 
- Place into `TabulatedCTG/`.

Download the CTU-CHB data: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Place waveform .dat files into `dat/`.

<a name="repro"></a>
## 3. Reproduction

### Machine Learning on Tabulated Dataset

Comparing different machine learning models with various physiology signals to assess the status of fetuses.
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/d7c0f112-893e-4353-8edd-fc59dad33bab" width="50%" height="50%">

To run ML methods on fetal_health.csv: `
python TabulatedCTG/MLMethods.py
`

```python
# Select columns for ML methods
df = df[['baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations',
        'severe_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability',
        'mean_value_of_short_term_variability', 'percentage_of_time_with_abnormal_long_term_variability',
        'mean_value_of_long_term_variability', 'fetal_health']]

# Create X and Y data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split the data into 75% training and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output the accuracies
print(model)
print(classification_report(y_test, y_pred))
print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred))

# Output the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1,2,3])
ConfusionMatrixDisplay(cm).plot()
```

- DecisionTree:
```
              precision    recall  f1-score   support

         1.0       0.95      0.95      0.95       418
         2.0       0.72      0.78      0.75        67
         3.0       0.90      0.81      0.85        47

    accuracy                           0.92       532
   macro avg       0.86      0.85      0.85       532
weighted avg       0.92      0.92      0.92       532

Balanced Accuracy:  0.8463918319428005
```
<img src="
https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/ccb22ec3-5691-442b-bcf5-91c7c385ca2b" width="50%" height="50%">

- GaussianNB:
```
GaussianNB()
              precision    recall  f1-score   support

         1.0       0.97      0.86      0.91       416
         2.0       0.49      0.86      0.63        79
         3.0       0.69      0.49      0.57        37

    accuracy                           0.83       532
   macro avg       0.72      0.74      0.70       532
weighted avg       0.88      0.83      0.84       532

Balanced Accuracy:  0.7351396856934831
```
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/a85a3fd1-cdfc-47fc-ae8d-56d5177347c8" width="50%" height="50%">

- RandomForest:
```
RandomForestClassifier(n_estimators=200)
              precision    recall  f1-score   support

         1.0       0.95      0.99      0.97       416
         2.0       0.89      0.72      0.79        78
         3.0       0.95      0.97      0.96        38

    accuracy                           0.95       532
   macro avg       0.93      0.89      0.91       532
weighted avg       0.94      0.95      0.94       532

Balanced Accuracy:  0.8924032838506523

```
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/a14e88a5-8acd-49c4-857a-5cfc510dc7f1" width="50%" height="50%">

- KNeighbors Model:
```
              precision    recall  f1-score   support

         1.0       0.91      0.96      0.94       415
         2.0       0.78      0.61      0.68        69
         3.0       0.85      0.71      0.77        48

    accuracy                           0.89       532
   macro avg       0.85      0.76      0.80       532
weighted avg       0.89      0.89      0.89       532

Balanced Accuracy:  0.7602948023979978
```
- Feature importance:
Using Random Forest model to calculate the feature importance:
```
features = ['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations','prolongued_decelerations','abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability','histogram_width','histogram_min','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes','histogram_mode','histogram_mean','histogram_median','histogram_variance','histogram_tendency']
f_i = list(zip(features,rf.feature_importances_))
f_i.sort(key = lambda x : x[1])
plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
plt.show()
```
The result clearly shows that `proloungued deceleration` and `uterine contractions` are important features to be considered about
![image](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/2c44bf3f-631c-4b70-8cda-b7646db4131a)

### CTU-CHB
#### Preprocessing

Analysis of fetal statuses with the [Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

- Download dataset: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Move dataset: `mv physionet.org/files/ctu-uhb-ctgdb/1.0.0/* CTU-CHB/dat/`
- Run `python3 waveform_processing.py` on dataset downloaded into `/dat`

Get the FHR signals by processing the CTU-CHB data.

Convert the signals to waveform:
```
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
```

#### Image Generation

- Recurrence Plots: `python3 CTU-CHB/generate_plots/generate_recurrence_plots.py`

The recurrence plot function used here:

```python
# Based on: https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def rec_plot(data, eps=0.3, steps=15):
    d = pdist(data[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z
```

Spectrograms and CWT:
- In matlab, run `generate_spectrograms.m` or `generate_cwts.m`

#### DCNN

Training: This assumes the data has been preprocessed, image plots have been generated, and you are in the `CTU-CHB/DCNN` directory:
- `python process_load_label.py` - Process, load and label 
- `python generate.py <imagetype>` - Generate test/train split: (where imagetype is either `spectrogram`, `cwt`, or `recurrence_plots`)
- `python train.py <num of epochs>` - Train Model:, 10-50 epochs recommended

To load model (which is seperated into architecture and weights):
```
with open('model_architecture.json', 'r') as json_file:
    architecture = json.load(json_file)
model = tf.keras.models.model_from_json(architecture)
model.load_weights('model.weights.h5')
```

