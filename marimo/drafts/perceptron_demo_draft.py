import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.datasets import make_blobs
    import polars as pl
    import numpy as np
    return go, make_blobs, mo, np, pl, px


@app.cell
def _(mo):
    center_x1 = mo.ui.slider(0, 10, 0.5)
    center_x1
    return (center_x1,)


@app.cell
def _(center_x1, np):
    # Parameters
    blob_centers = [[center_x1.value, 3], [10, 5]]
    blob_std = 0.1
    x1_range = [0, 10]
    x2_range = [0, 10]
    nx = 50
    x1_ax = np.linspace(start=x1_range[0], stop=x1_range[1], endpoint=False)
    x2_ax = np.linspace(start=x2_range[0], stop=x2_range[1], endpoint=False)
    x1_grid, x2_grid = np.meshgrid(x1_ax, x2_ax)
    image = x1_grid + x2_grid
    return blob_centers, image, nx, x1_ax, x1_range, x2_ax, x2_range


@app.cell
def _(image):
    print(image.shape)
    return


@app.cell
def _(X, go, image, mo, px, x1_ax, x2_ax):
    # im_plot = mo.ui.plotly(
    #     px.imshow(
    #         image,
    #         x=x1_ax,
    #         y=x2_ax,
    #         origin="lower",
    #         color_continuous_scale="Portland",
    #         color_continuous_midpoint=10,
    #     )
    # )
    # im_plot
    fig1 = px.imshow(
        image,
        x=x1_ax,
        y=x2_ax,
        origin="lower",
        color_continuous_scale="Portland",
        color_continuous_midpoint=10,
    )
    fig1.add_trace(
        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", xaxis="x", yaxis="y")
    )
    # fig1.show()
    mo.ui.plotly(fig1)
    return


@app.cell
def _(blob_centers, make_blobs, np, pl):
    X, y = make_blobs(centers=blob_centers, random_state=42)
    y = np.astype(y, str)  # String indicates categorical
    df = pl.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "class": y})
    return (X,)


@app.cell
def _(X, go, image, nx, x1_range, x2_range):
    fig = go.Figure()
    # fig = px.scatter(df, x="x1", y="x2", color="class", range_x=x1_range, range_y=x2_range)
    # fig.add_trace(
    #     go.Image(
    #         z=image,
    #     )
    # )
    fig.add_trace(
        go.Heatmap(
            z=image,
            xaxis="x",
            yaxis="y",
            dx=x1_range[1] / nx,
            dy=x2_range[1] / nx,
            x0=0,
            y0=0,
            colorscale="Portland",
        )
    )

    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", xaxis="x", yaxis="y"))

    # 6. Configure layout
    fig.update_layout(
        title="Scatter Plot Overlay on Raster Data",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
        xaxis_range=x1_range,
        yaxis_range=x2_range,
    )

    # 7. Show the plot
    fig.show()
    # # Reverse y-axis so scatter aligns with image
    # fig.update_yaxes(autorange="reversed")

    # # Clean up axes
    # fig.update_xaxes(showgrid=False, zeroline=False)
    # fig.update_yaxes(showgrid=False, zeroline=False)

    # # Show in marimo
    # mo.ui.plotly(fig)
    # # plot = mo.ui.plotly(_plot)
    return


@app.cell
def _():
    #
    return


@app.cell
def _(go, image, mo, nx, x1_range, x2_range):
    f2 = mo.ui.plotly(
        go.Image(
            z=image,
            dx=x1_range[1] / nx,
            dy=x2_range[1] / nx,
            x0=0,
            y0=0,
        )
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
