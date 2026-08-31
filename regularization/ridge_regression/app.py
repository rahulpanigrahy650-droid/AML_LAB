from flask import Flask, render_template, request
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

GRAPH_PATH = "static/regression.png"
COST_GRAPH_PATH = "static/cost.png"


def calculate_metrics(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(mse)

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)

    r2 = 0 if ss_total == 0 else 1 - (ss_residual / ss_total)

    return mse, mae, rmse, r2


def ridge_gradient_descent(x, y, w, b, learning_rate, iterations, alpha):
    """
    Ridge (L2) Regression using Gradient Descent.
    Cost = MSE + alpha * w^2
    """

    n = len(x)
    history = []

    for i in range(1, iterations + 1):

        y_pred = w * x + b

        dw = (-2 / n) * np.sum(x * (y - y_pred)) + 2 * alpha * w
        db = (-2 / n) * np.sum(y - y_pred)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        y_pred = w * x + b
        mse, mae, rmse, r2 = calculate_metrics(y, y_pred)

        cost = mse + alpha * (w ** 2)  # MSE + L2 penalty

        history.append({
            "iteration": i,
            "weight": w,
            "bias": b,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cost": cost
        })

    return w, b, history


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        action = request.form.get("action", "calculate")

        try:
            x = np.array(list(map(float, request.form["x_values"].split(","))))
            y = np.array(list(map(float, request.form["y_values"].split(","))))

            w = float(request.form["weight"])
            b = float(request.form["bias"])

            learning_rate = float(request.form["learning_rate"])
            iterations = int(request.form["iterations"])
            alpha = float(request.form["alpha"])
        except ValueError:
            return render_template(
                "index.html",
                error="Please check your inputs. X/Y must be comma-separated numbers."
            )

        if len(x) != len(y):
            return render_template(
                "index.html",
                error="X and Y must contain the same number of values."
            )

        if len(x) < 2:
            return render_template(
                "index.html",
                error="Please provide at least 2 data points."
            )

        final_w, final_b, history = ridge_gradient_descent(
            x, y, w, b, learning_rate, iterations, alpha
        )

        y_pred = final_w * x + final_b
        mse, mae, rmse, r2 = calculate_metrics(y, y_pred)

        show_graphs = (action == "visualize")

        if show_graphs:

            # GRAPH 1 - Regression Line
            plt.figure(figsize=(7, 5))
            plt.scatter(x, y, label="Original Data")

            sort_index = np.argsort(x)
            plt.plot(x[sort_index], y_pred[sort_index], color="red", label="Ridge Regression Line")

            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Ridge Regression with Gradient Descent")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(GRAPH_PATH)
            plt.close()

            # GRAPH 2 - Cost Function vs Iterations
            iterations_list = [item["iteration"] for item in history]
            cost_values = [item["cost"] for item in history]

            plt.figure(figsize=(7, 5))
            plt.plot(iterations_list, cost_values, color="steelblue")

            plt.xlabel("Number of Iterations")
            plt.ylabel("Cost (MSE + L2 Regularization)")
            plt.title("Cost History during Gradient Descent")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(COST_GRAPH_PATH)
            plt.close()

        return render_template(
            "index.html",
            final_w=final_w,
            final_b=final_b,
            equation=f"y = {final_w:.4f}x + {final_b:.4f}",
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2=r2,
            history=history,
            show_results=True,
            show_graphs=show_graphs
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5052, debug=True, use_reloader=False)
