import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("TabulatedCTG/fetal_health.csv")
print(df.describe())