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


def compute_regression(x, y):
    """Runs the Least Squares Method and returns slope, intercept, and error metrics."""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_x2) - (sum_x ** 2))
    intercept = (sum_y - slope * sum_x) / n

    y_pred = slope * x + intercept
    residuals = y - y_pred

    mae = np.mean(np.abs(residuals))
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return {
        "n": n, "sum_x": sum_x, "sum_y": sum_y, "sum_xy": sum_xy, "sum_x2": sum_x2,
        "slope": slope, "intercept": intercept,
        "mae": mae, "mse": mse, "rmse": rmse, "r2": r2,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    show_graph = False
    x_input = y_input = ""

    if request.method == "POST":
        action = request.form.get("action", "calculate")
        x_input = request.form.get("x_values", "")
        y_input = request.form.get("y_values", "")

        try:
            x = np.array([float(v) for v in x_input.split(",")])
            y = np.array([float(v) for v in y_input.split(",")])

            if len(x) != len(y):
                raise Exception("X and Y must have the same number of values.")
            if len(x) < 2:
                raise Exception("Enter at least two data points.")

            r = compute_regression(x, y)

            if action == "visualize":
                # ---- Scatter plot + regression line + metrics as text on the plot ----
                plt.figure(figsize=(7, 5))
                plt.scatter(x, y, color="#4361ee", s=70, label="Data Points", edgecolor="white")

                x_line = np.linspace(x.min(), x.max(), 100)
                plt.plot(x_line, r["slope"] * x_line + r["intercept"], color="#f72585",
                          linewidth=2.5, label="Regression Line")

                metrics_text = (
                    f"MAE  = {r['mae']:.4f}\n"
                    f"MSE  = {r['mse']:.4f}\n"
                    f"RMSE = {r['rmse']:.4f}\n"
                    f"R\u00b2   = {r['r2']:.4f}"
                )
                plt.text(
                    0.02, 0.98, metrics_text,
                    transform=plt.gca().transAxes,
                    fontsize=10, fontfamily="monospace",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="#4361ee", alpha=0.9)
                )

                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title("Linear Regression \u2014 Least Squares Method")
                plt.grid(alpha=0.3)
                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig(GRAPH_PATH, dpi=110)
                plt.close()

                show_graph = True

            else:
                result = {
                    "n": r["n"],
                    "sum_x": round(r["sum_x"], 4),
                    "sum_y": round(r["sum_y"], 4),
                    "sum_xy": round(r["sum_xy"], 4),
                    "sum_x2": round(r["sum_x2"], 4),
                    "slope": round(r["slope"], 4),
                    "intercept": round(r["intercept"], 4),
                    "equation": f"y = {round(r['slope'], 4)}x + {round(r['intercept'], 4)}",
                    "mae": round(r["mae"], 4),
                    "mse": round(r["mse"], 4),
                    "rmse": round(r["rmse"], 4),
                    "r2": round(r["r2"], 4),
                }

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error, show_graph=show_graph,
                            x_input=x_input, y_input=y_input)


if __name__ == "__main__":
    app.run(debug=True)
