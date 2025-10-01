# Imports
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Read noisy heart rate data
df = pl.read_csv("jupyter/noisy_heart_rate_time_series.csv")
time = df["t"].to_numpy()
hr_noisy = df["heart_rate_noisy"].to_numpy()
hr_accurate = df["heart_rate_accurate"].to_numpy()
w1 = 0.9
w2 = 0.01
b = 0
learning_rate = 0.001
output = []

y_prev = 0.0
for x, y in zip(hr_noisy, hr_accurate):
    y_pred = w1 * x + w2 * y_prev + b
    output.append(y_pred)
    error = y - y_pred
    # loss = error**2
    update = learning_rate * 2 * error
    w1 += update * x
    w2 += update * y_prev
    b += update * 1.0
    y_prev = y_pred

print(output)
