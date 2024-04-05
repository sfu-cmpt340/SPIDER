
# Fetal Health Classification from Cardiotocography Data![download](https://github.com/sfu-cmpt340/fetal-health-classification/assets/59947126/559234c8-60a4-4a66-862d-5a1c8b7c89cf)


# Introduction ✏️
Using this [dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data), which has cardiotocography (CTG) biosignals for 2126 fetuses, each with 22 features and a label: Normal, Suspect, and Pathological.
Our goal is to create a robust model which can predict fetal health from this data.
We plan to train and test several classification models (such as linear, svm, supervised learning ML) to predict/classify the health of the fetus, and identify which model is most effective.

## Important Links 🔗

| [Dataset Download](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data) | [Slack channel](https://app.slack.com/client/T06AP91EYG6/C06DW38TA3X) | [Project report](https://www.overleaf.com/project/65a57b95a9883102c00a9e4b) | [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EXfKWdGF-QBCtgFGivjFPycBfwZrCIoGwRcEkODk1pRWkw?e=M1zm8O) |



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
### 1) Machine Learning Demo

Comparing different machine learning models with various physiology signals to assess the status of fetuses.
![Figure_1](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/d7c0f112-893e-4353-8edd-fc59dad33bab)


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

### DCNN
Model for graph-structured data, using recurrence plots, spectrograms and cwt as inputs.
Generating the classification results based on fetal heart rate signals.
- Install requirements first: `pip install -r requirements.txt`
- Run the test on the prepared dataset: `python test.py`
```python
predictions = model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)
class_counts = np.bincount(predicted_classes)

print("Count of guesses in each class:")
for cls, count in enumerate(class_counts):
    print(f"Class {cls}: {count} guesses")
```


### Recurrence Plots
- Download dataset: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Feature extraction: `python3 "Feature Extraction.py"`
- Run the program: `python3 GenerateRecurrencePlots.py`

The recurrece plot function used here:

```python
# Based on: https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def rec_plot(data, eps=0.3, steps=15):
    d = pdist(data[:, None])
    d = np.floor(d / eps)
    d[d > steps] = steps
    Z = squareform(d)
    return Z
```
Here is one sample recurrece plot:

![image](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/f5d9582d-7793-4a3f-bafe-6b67b58a3331)


### Spectrogram
Get the FHR signals by processing the CTU-CHB data.
Convert the signals to waveform:
- Install wfdb: `! pip install wfdb`
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
Generate the spectrogram with matlab:
`[S, f, t] = spectrogram(waveform, window_size, overlap, nfft);`


### What to find where

Explain briefly what files are found where

```bash
repository
├── Machine Learning Method Testing      ## Comparison of different kinds of machine learning models
├── CTU-CHB                              ## Work on CTU-CHB database
├── README.md                            ## Introduction of the project
├── Tabulate data.ipynb                  ## To tabulate the data 
├── requirements.yml                     ## If you use conda
```

<a name="installation"></a>

## 2. Installation

### DCNN
- Install requirements: `pip install -r requirements.txt`
- Process, load and label: `python process_load_label.py`
- Generate test/train split (takes a min or more): `python generate.py <imagetype>` (where imagetype is either `spectrogram`, `cwt`, or `recurrence_plots`)
- Train Model: `python train.py <num of epochs>`, 10-50 epochs recommended

Sample trained models are in the folder 'Pre-trained Samples'. Run `python test.py` to test these models.


To load and use: 

```
with open('model_architecture.json', 'r') as json_file:
    architecture = json.load(json_file)
model = tf.keras.models.model_from_json(architecture)
model.load_weights('model.weights.h5')
```

### Recurrence Plots
- Download dataset: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Feature extraction: `python3 "Feature Extraction.py"`
- Run the program: `python3 GenerateRecurrencePlots.py`

### Spectrogram
- Download dataset: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`
- Feature extraction: `python3 "Feature Extraction.py"`
- Run the program: `matlab -nodisplay -nosplash -nodesktop -r "run('generate_spectrograms.m');exit;"`


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
![dt](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/ccb22ec3-5691-442b-bcf5-91c7c385ca2b)

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
![GNB](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/a85a3fd1-cdfc-47fc-ae8d-56d5177347c8)

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
![rf](https://github.com/sfu-cmpt340/fetal-health-classification/assets/113268694/a14e88a5-8acd-49c4-857a-5cfc510dc7f1)

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
By analysing the graph data generated from the CTU-CHB dataset's signals, we can classify the classes of fetals:
```
5/5 ━━━━━━━━━━━━━━━━━━━━ 0s 43ms/step
Class 0: 124 guesses
Class 1: 10 guesses
Accuracy: 0.7835820895522388
Recall: 0.546036690896504
F1-score: 0.5453375453375453
Precision: 0.6072580645161291
[[101   6]
 [ 23   4]]
```

## Data Downloading

Download the `fetal-health.csv`: https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/data
Download the CTU-CHB data: `wget -r -N -c -np https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/`


