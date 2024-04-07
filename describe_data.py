import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("TabulatedCTG/fetal_health.csv")
print(df['fetal_health'].value_counts())

df2 = pd.read_csv("CTU-CHB/outcomes.csv")
print(df2.head())

dcnn_res = [
    "accuracy: 0.8154 - f1_score: 0.7772 - loss: 0.5842 - precision: 0.8154 - recall: 0.8154 - val_accuracy: 0.7100 - val_f1_score: 0.4738 - val_loss: 0.8073 - val_precision: 0.7100 - val_recall: 0.7100"
]

def parse_metrics(data):
    metrics = {}
    for item in data:
        parts = item.split(' - ')
        for part in parts:
            key, value = part.split(': ')
            if key in metrics:
                metrics[key].append(float(value))
            else:
                metrics[key] = [float(value)]
    return metrics

data = [
    "accuracy: 0.8684 - f1_score: 0.8337 - loss: 0.5645 - precision: 0.8684 - recall: 0.8684 - val_accuracy: 0.7200 - val_f1_score: 0.4792 - val_loss: 0.7811 - val_precision: 0.7200 - val_recall: 0.7200",
    "accuracy: 0.8099 - f1_score: 0.7726 - loss: 0.6422 - precision: 0.8099 - recall: 0.8099 - val_accuracy: 0.7500 - val_f1_score: 0.4959 - val_loss: 0.7857 - val_precision: 0.7500 - val_recall: 0.7500",
    "accuracy: 0.8229 - f1_score: 0.7829 - loss: 0.6495 - precision: 0.8229 - recall: 0.8229 - val_accuracy: 0.7700 - val_f1_score: 0.4743 - val_loss: 0.8445 - val_precision: 0.7700 - val_recall: 0.7700",
    "accuracy: 0.8567 - f1_score: 0.8178 - loss: 0.6016 - precision: 0.8567 - recall: 0.8567 - val_accuracy: 0.8000 - val_f1_score: 0.4893 - val_loss: 0.7473 - val_precision: 0.8000 - val_recall: 0.8000",
    "accuracy: 0.8038 - f1_score: 0.7477 - loss: 0.7341 - precision: 0.8038 - recall: 0.8038 - val_accuracy: 0.7300 - val_f1_score: 0.4846 - val_loss: 0.7913 - val_precision: 0.7300 - val_recall: 0.7300",
    
    "accuracy: 0.7999 - f1_score: 0.7573 - loss: 0.7144 - precision: 0.7999 - recall: 0.7999 - val_accuracy: 0.7600 - val_f1_score: 0.5748 - val_loss: 0.8098 - val_precision: 0.7600 - val_recall: 0.7600",
    "accuracy: 0.7453 - f1_score: 0.7056 - loss: 0.7236 - precision: 0.7453 - recall: 0.7453 - val_accuracy: 0.6400 - val_f1_score: 0.5434 - val_loss: 0.8770 - val_precision: 0.6400 - val_recall: 0.6400",
    "accuracy: 0.7979 - f1_score: 0.7468 - loss: 0.6686 - precision: 0.7979 - recall: 0.7979 - val_accuracy: 0.7600 - val_f1_score: 0.5294 - val_loss: 0.7541 - val_precision: 0.7600 - val_recall: 0.7600",
    "accuracy: 0.8034 - f1_score: 0.7632 - loss: 0.6786 - precision: 0.8034 - recall: 0.8034 - val_accuracy: 0.7400 - val_f1_score: 0.4902 - val_loss: 0.7803 - val_precision: 0.7400 - val_recall: 0.7400",
    "accuracy: 0.8154 - f1_score: 0.7642 - loss: 0.6398 - precision: 0.8154 - recall: 0.8154 - val_accuracy: 0.7500 - val_f1_score: 0.6018 - val_loss: 0.7391 - val_precision: 0.7500 - val_recall: 0.7500",
    
    "accuracy: 0.7809 - f1_score: 0.7355 - loss: 0.7197 - precision: 0.7809 - recall: 0.7809 - val_accuracy: 0.5400 - val_f1_score: 0.5066 - val_loss: 0.8927 - val_precision: 0.5400 - val_recall: 0.5400",
    "accuracy: 0.7030 - f1_score: 0.6499 - loss: 0.9283 - precision: 0.7030 - recall: 0.7030 - val_accuracy: 0.3300 - val_f1_score: 0.3299 - val_loss: 0.9912 - val_precision: 0.3300 - val_recall: 0.3300",
    "accuracy: 0.7155 - f1_score: 0.6663 - loss: 0.7355 - precision: 0.7155 - recall: 0.7155 - val_accuracy: 0.5800 - val_f1_score: 0.5091 - val_loss: 0.9129 - val_precision: 0.5800 - val_recall: 0.5800",
    "accuracy: 0.7575 - f1_score: 0.7058 - loss: 0.7474 - precision: 0.7575 - recall: 0.7575 - val_accuracy: 0.6900 - val_f1_score: 0.5408 - val_loss: 0.8244 - val_precision: 0.6900 - val_recall: 0.6900",
    "accuracy: 0.7690 - f1_score: 0.7132 - loss: 0.8580 - precision: 0.7690 - recall: 0.7690 - val_accuracy: 0.6600 - val_f1_score: 0.4241 - val_loss: 0.8440 - val_precision: 0.6600 - val_recall: 0.6600",
]   

metrics = parse_metrics(data)

df = pd.DataFrame(metrics)
df.to_excel("metrics.xlsx", index=False)