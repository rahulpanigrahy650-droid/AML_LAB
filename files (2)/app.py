from flask import Flask, render_template, request

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os


app = Flask(__name__)


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

GRAPH_PATH = os.path.join(STATIC_DIR, "graph.png")
METRIC_GRAPH_PATH = os.path.join(STATIC_DIR, "metrics.png")


def compute_regression(x, y):

    n = len(x)

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    slope = ((n * sum_xy) - (sum_x * sum_y)) / (
        (n * sum_x2) - (sum_x ** 2)
    )

    intercept = (sum_y - slope * sum_x) / n


    y_pred = slope * x + intercept

    residuals = y - y_pred


    mae = np.mean(np.abs(residuals))

    mse = np.mean(residuals ** 2)

    rmse = np.sqrt(mse)


    ss_res = np.sum(residuals ** 2)

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


    return {

        "n": n,

        "sum_x": sum_x,

        "sum_y": sum_y,

        "sum_xy": sum_xy,

        "sum_x2": sum_x2,

        "slope": slope,

        "intercept": intercept,

        "mae": mae,

        "mse": mse,

        "rmse": rmse,

        "r2": r2

    }



@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    error = None

    show_graph = False

    action = "calculate"


    x_input = ""

    y_input = ""



    if request.method == "POST":


        action = request.form.get(
            "action",
            "calculate"
        )


        x_input = request.form.get(
            "x_values",
            ""
        )


        y_input = request.form.get(
            "y_values",
            ""
        )


        try:


            x = np.array(
                [
                    float(v)
                    for v in x_input.split(",")
                ]
            )


            y = np.array(
                [
                    float(v)
                    for v in y_input.split(",")
                ]
            )


            if len(x) != len(y):

                raise Exception(
                    "X and Y must have same number of values"
                )


            if len(x) < 2:

                raise Exception(
                    "Enter at least two data points"
                )



            r = compute_regression(
                x,
                y
            )


            result = {


                "equation":
                f"y = {r['slope']:.4f}x + {r['intercept']:.4f}",


                "n": r["n"],


                "sum_x":
                round(r["sum_x"],4),


                "sum_y":
                round(r["sum_y"],4),


                "sum_xy":
                round(r["sum_xy"],4),


                "sum_x2":
                round(r["sum_x2"],4),


                "slope":
                round(r["slope"],4),


                "intercept":
                round(r["intercept"],4),


                "mae":
                round(r["mae"],4),


                "mse":
                round(r["mse"],4),


                "rmse":
                round(r["rmse"],4),


                "r2":
                round(r["r2"],4)

            }



            if action == "visualize":



                # Regression Scatter Plot


                plt.figure(figsize=(7,5))


                plt.scatter(
                    x,
                    y,
                    color="#4361ee",
                    s=70,
                    edgecolor="white",
                    label="Data Points"
                )


                x_line = np.linspace(
                    x.min(),
                    x.max(),
                    100
                )


                plt.plot(
                    x_line,
                    r["slope"] * x_line + r["intercept"],
                    color="#f72585",
                    linewidth=2.5,
                    label="Regression Line"
                )


                plt.xlabel("X")

                plt.ylabel("Y")

                plt.title(
                    "Linear Regression Scatter Plot"
                )

                plt.grid(alpha=0.3)

                plt.legend()

                plt.tight_layout()


                plt.savefig(
                    GRAPH_PATH,
                    dpi=110
                )

                plt.close()




                # Metrics Scatter Plot


                plt.figure(figsize=(7,5))


                metrics = [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R²"
                ]


                values = [
                    r["mae"],
                    r["mse"],
                    r["rmse"],
                    r["r2"]
                ]


                positions = np.arange(
                    len(metrics)
                )


                plt.scatter(
                    positions,
                    values,
                    color="red",
                    s=150
                )


                plt.xticks(
                    positions,
                    metrics
                )


                plt.xlabel(
                    "Evaluation Metrics"
                )


                plt.ylabel(
                    "Value"
                )


                plt.title(
                    "Evaluation Metrics Scatter Plot"
                )


                plt.grid(
                    alpha=0.3
                )


                for i, value in enumerate(values):

                    plt.text(
                        positions[i],
                        value,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom"
                    )



                plt.tight_layout()


                plt.savefig(
                    METRIC_GRAPH_PATH,
                    dpi=110
                )


                plt.close()



                show_graph = True



        except Exception as e:

            error = str(e)



    return render_template(

        "index.html",

        result=result,

        error=error,

        show_graph=show_graph,

        action=action,

        x_input=x_input,

        y_input=y_input

    )



if __name__ == "__main__":

    app.run(debug=True)