# Imports
import numpy as np
import plotly.graph_objects as go
import polars as pl

# Parameters
w1 = 55
w2 = 1.0
b = -100

x1_range = (0, 3.5)
x2_range = (30, 200)

x1_ax = np.linspace(start=x1_range[0], stop=x1_range[1])
x2_ax = np.linspace(start=x2_range[0], stop=x2_range[1])
x1_grid, x2_grid = np.meshgrid(x1_ax, x2_ax)


# Read data from CSV file
df = pl.read_csv("jupyter/simple_acc_hr_dataset.csv")

X = df.select(pl.col(["acceleration (m/s2)", "heart_rate (bpm)"])).to_numpy()
y = df.select(pl.col("state_int")).to_numpy().flatten()

# Use weights to calculate "logit" z for points and for grid
z = X @ np.array([w1, w2]) + b
y_pred = z > 0
z_grid = x1_grid * w1 + x2_grid * w2 + b

# Create heatmap for perceptron output (z)
heatmap = go.Heatmap(
    x=x1_ax,
    y=x2_ax,
    z=z_grid,
    colorscale="BuRd",
    zmin=-50,
    zmax=50,
    opacity=0.4,
    colorbar=dict(title="Perceptron output (z)", title_side="right"),
)

exercise_correct = (y == 1) & (y_pred == 1)
rest_correct = (y == 0) & (y_pred == 0)
exercise_incorrect = (y == 1) & (y_pred == 0)
rest_incorrect = (y == 0) & (y_pred == 1)


# Create scatter plots for each category
scatter_exercise_correct = go.Scatter(
    x=X[exercise_correct, 0],
    y=X[exercise_correct, 1],
    mode="markers",
    marker=dict(symbol="circle", color="red"),
    name="exercise correct",
)
scatter_rest_correct = go.Scatter(
    x=X[rest_correct, 0],
    y=X[rest_correct, 1],
    mode="markers",
    marker=dict(symbol="circle", color="blue"),
    name="rest correct",
)
scatter_exercise_incorrect = go.Scatter(
    x=X[exercise_incorrect, 0],
    y=X[exercise_incorrect, 1],
    mode="markers",
    marker=dict(symbol="x", color="red", size=12),
    name="exercise misclass.",
)
scatter_rest_incorrect = go.Scatter(
    x=X[rest_incorrect, 0],
    y=X[rest_incorrect, 1],
    mode="markers",
    marker=dict(symbol="x", color="blue", size=12),
    name="rest misclass.",
)

# Combine all traces
fig = go.Figure(
    data=[
        heatmap,
        scatter_exercise_correct,
        scatter_rest_correct,
        scatter_exercise_incorrect,
        scatter_rest_incorrect,
    ]
)
fig.update_layout(
    xaxis_title="acceleration (m/s2)",
    yaxis_title="heart_rate (bpm)",
    legend_title="Legend",
    legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
    xaxis=dict(range=[x1_range[0], x1_range[1]]),
    yaxis=dict(range=[x2_range[0], x2_range[1]]),
)
fig.show()
