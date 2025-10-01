import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Imports
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import marimo as mo
    return go, mo, np, pl


@app.cell
def _(mo):
    # Parameters
    w1 = mo.ui.slider(-50, 50, 10)
    w2 = mo.ui.slider(-5, 5, 0.1)
    b = mo.ui.slider(-100, 100, 10)

    x1_range = (0, 3.5)
    x2_range = (30, 200)

    z_range = (-50, 50)
    return b, w1, w2, x1_range, x2_range, z_range


@app.cell
def _(pl):
    # Read data from CSV file
    df = pl.read_csv("marimo/simple_acc_hr_dataset.csv")

    X = df.select(pl.col(["acceleration (m/s2)", "heart_rate (bpm)"])).to_numpy()
    y = df.select(pl.col("state_int")).to_numpy().flatten()
    return X, y


@app.cell
def _(X, b, np, w1, w2, x1_grid, x2_grid):
    # Use weights to calculate "logit" z for points and for grid
    z = X @ np.array([w1.value, w2.value]) + b.value
    y_pred = z > 0
    z_grid = x1_grid * w1.value + x2_grid * w2.value + b.value
    return y_pred, z_grid


@app.cell
def _(y, y_pred):
    # Evaluate accuracy
    exercise_correct = (y == 1) & (y_pred == 1)
    rest_correct = (y == 0) & (y_pred == 0)
    exercise_incorrect = (y == 1) & (y_pred == 0)
    rest_incorrect = (y == 0) & (y_pred == 1)
    return exercise_correct, exercise_incorrect, rest_correct, rest_incorrect


@app.cell
def _(np, x1_range, x2_range):
    x1_ax = np.linspace(start=x1_range[0], stop=x1_range[1])
    x2_ax = np.linspace(start=x2_range[0], stop=x2_range[1])
    x1_grid, x2_grid = np.meshgrid(x1_ax, x2_ax)
    return x1_ax, x1_grid, x2_ax, x2_grid


@app.cell
def _(b, mo, w1, w2):
    mo.vstack([w1, w2, b])
    return


@app.cell(hide_code=True)
def _(
    X,
    exercise_correct,
    exercise_incorrect,
    go,
    mo,
    rest_correct,
    rest_incorrect,
    x1_ax,
    x1_range,
    x2_ax,
    x2_range,
    z_grid,
    z_range,
):
    # Create heatmap for perceptron output (z)
    heatmap = go.Heatmap(
        x=x1_ax,
        y=x2_ax,
        z=z_grid,
        colorscale="RdBu_r",
        zmin=z_range[0],
        zmax=z_range[1],
        opacity=0.4,
        colorbar=dict(title="Perceptron output (z)", title_side="right"),
    )


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
        yaxis_title="heart rate (bpm)",
        legend_title="Legend",
        legend=dict(x=1, y=0, xanchor="right", yanchor="bottom"),
        xaxis=dict(range=[x1_range[0], x1_range[1]]),
        yaxis=dict(range=[x2_range[0], x2_range[1]]),
    )
    # fig.show()
    mo.ui.plotly(fig)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
