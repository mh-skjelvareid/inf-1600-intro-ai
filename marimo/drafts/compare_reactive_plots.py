import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    import marimo as mo

    return mo, np, pd, plt


@app.cell
def _(mo):
    # Slider input
    slider = mo.ui.slider(1, 20, value=5, label="Frequency")
    return (slider,)


@app.cell
def _(np, pd, slider):
    freq = slider.value
    x = np.linspace(0, 10, 500)
    y = np.sin(freq * x)
    df = pd.DataFrame({"x": x, "y": y})
    return freq, x, y


@app.cell
def _(slider):
    slider
    return


@app.cell
def _():
    # # Nice plot but flickers on update
    # mo.ui.altair_chart(alt.Chart(df).mark_line().encode(x="x", y="y"))
    return


@app.cell
def _():
    # THIS WORKS BEST!!
    mo.ui.plotly(go.Figure(data=go.Scatter(x=x, y=y, mode="lines")))
    return


@app.cell
def _(freq, mo, plt, x, y):
    # # Quite cluncky
    # fig, ax = plt.subplots()
    # ax.plot(x, y)
    # ax.set_title(f"Frequency = {freq}")
    # plt.show()
    # mo.mpl.interactive(fig)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
