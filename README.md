
# Fetal Health Classification from Cardiotocography Data![download](https://github.com/sfu-cmpt340/fetal-health-classification/assets/59947126/559234c8-60a4-4a66-862d-5a1c8b7c89cf)


# Introduction ✏️
Using this [tabulated dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data), which has cardiotocography (CTG) biosignals for 2126 fetuses, each with 22 features and a label: Normal, Suspect, and Pathological.
[CTU-CHB dataset](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

Our goal is to create a robust model which can predict fetal health from this data.
We plan to train and test several classification models (such as linear, svm, supervised learning ML) to predict/classify the health of the fetus, and identify which model is most effective.

## Important Links 🔗

| [Tabulated Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data) | [CTU-CHB dataset](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW38TA3X) | [Project report](https://www.overleaf.com/project/65a57b95a9883102c00a9e4b) | [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EXfKWdGF-QBCtgFGivjFPycBfwZrCIoGwRcEkODk1pRWkw?e=M1zm8O) |

- Dataset Download: Link to download the dataset of this project.
- Slack channel: Link to private Slack project channel.
- Project report: Link to Overleaf project report document.
- Timesheet: Link to timesheet (pinned in your project's Slack channel) where you track per student the time and tasks completed/participated for this project/


## Video/demo/GIF 📽️
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)


<a name="demo"></a>
## 1. Demo 📝

### Layout Overview

```bash
repository
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
├── TabulatedCTG                ## Tabulated CTG dataset
│   ├── Classify CTU-CHB            ## ML methods on CTU-CHB
│   ├── fetal_health.csv            ## Tabulated Dataset
│   ├── ML Methods.py               ## Machine learning methods
│   ├── NN.py                       ## Neural Network classifier
├── requirements.txt            ## Setup requirements
```

### 1) Machine Learning Demo

Comparing different machine learning models with various physiology signals to assess the status of fetuses.
<img src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/d7c0f112-893e-4353-8edd-fc59dad33bab" width="50%" height="50%">


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

### 2) CTU-CHB Demo

Analysis of fetal statuses with the [Intrapartum Cardiotocography Database](https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/)

### Pre-processing
Get the FHR signals by processing the CTU-CHB data.
Convert the signals to waveform:
- Install wfdb: `! pip install wfdb`

- Run `python3 waveform_processing.py` on dataset downloaded into `/dat`

Example signal before and after preprocessing:

<img width="1049" alt="Screenshot 2024-04-06 at 1 27 25 PM" src="https://github.com/sfu-cmpt340/fetal-health-classification/assets/29849456/0449a66b-eedd-4462-9eec-8e17bbd00d40">

### DCNN
Model for graph-structured data, using recurrence plots, spectrograms and cwt as inputs.
Generating the classification results based on fetal heart rate signals.
- Run the test on the prepared dataset: `DCNN/python test.py`

### Recurrence Plots
- Download dataset: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Feature extraction: `python3 "Feature Extraction.py"`
- Run the program: `python3 generate_recurrence_plots.py`

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
Here is one sample recurrence plot:

![image](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/f5d9582d-7793-4a3f-bafe-6b67b58a3331)

<a name="installation"></a>

## 2. Installation

Install requirements: `pip install -r requirements.txt`

DCNN requires tensorflow to be setup.


<a name="repro"></a>
## 3. Reproduction

### Machine Learning
`python3 "ML Methods.py"`
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
Preprocess .dat waveform files.
#### Image Generation
Generate plots as images.
#### DCNN
- Process, load and label: `python process_load_label.py`
- Generate test/train split (takes a min or more): `python generate.py <imagetype>` (imagetype is either `spectrogram`, `cwt`, or `recurrence_plots`)
- Train Model: `python train.py <num of epochs>`, 15-30 epochs recommended

Sample trained models are in the folder 'Pre-trained Samples'. Run `python test.py` to test these models, output shown below:

```
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step
Class 0: 124 guesses
Class 1: 10 guesses
Accuracy: 0.7835820895522388
Recall: 0.546036690896504
F1-score: 0.5453375453375453
Precision: 0.6072580645161291
```

## Data Downloading

Download the `fetal-health.csv`: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data
Download the CTU-CHB data: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`


