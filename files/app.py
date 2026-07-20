from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import io
import base64

app = Flask(__name__)

# Global in-memory state (single-user demo app)
STATE = {
    "model": None,
    "feature_names": [],
    "X_train": None,
    "y_train": None,
    "y_pred_train": None,
    "target_name": None,
    "prediction": None,
    "metrics": None,
}


def reset_state():
    STATE["model"] = None
    STATE["feature_names"] = []
    STATE["X_train"] = None
    STATE["y_train"] = None
    STATE["y_pred_train"] = None
    STATE["target_name"] = None
    STATE["prediction"] = None
    STATE["metrics"] = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.get_json()

        feature_names = data["feature_names"]          # list of strings
        feature_values = data["feature_values"]         # dict: name -> list of floats
        target_name = data["target_name"]                # string
        target_values = data["target_values"]             # list of floats
        test_values = data["test_values"]                  # dict: name -> float

        # Build training DataFrame
        df = pd.DataFrame({name: feature_values[name] for name in feature_names})
        y = np.array(target_values, dtype=float)

        if len(df) < 2:
            return jsonify({"error": "Please provide at least 2 training rows for each feature and the target."}), 400

        row_lengths = {name: len(feature_values[name]) for name in feature_names}
        if len(set(row_lengths.values())) != 1 or list(row_lengths.values())[0] != len(target_values):
            return jsonify({"error": "All feature columns and the target column must have the same number of values."}), 400

        X = df.values

        model = LinearRegression()
        model.fit(X, y)

        y_pred_train = model.predict(X)

        r2 = r2_score(y, y_pred_train)
        mse = mean_squared_error(y, y_pred_train)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred_train)

        # Predict on the new test row
        test_row = np.array([[float(test_values[name]) for name in feature_names]])
        prediction = float(model.predict(test_row)[0])

        coefficients = {name: float(coef) for name, coef in zip(feature_names, model.coef_)}
        intercept = float(model.intercept_)

        # Save state for the /visualize endpoint
        STATE["model"] = model
        STATE["feature_names"] = feature_names
        STATE["X_train"] = X
        STATE["y_train"] = y
        STATE["y_pred_train"] = y_pred_train
        STATE["target_name"] = target_name
        STATE["prediction"] = prediction
        STATE["metrics"] = {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

        return jsonify({
            "prediction": round(prediction, 4),
            "metrics": {
                "r2": round(r2, 4),
                "mse": round(mse, 4),
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
            },
            "coefficients": {k: round(v, 4) for k, v in coefficients.items()},
            "intercept": round(intercept, 4),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/visualize", methods=["POST"])
def visualize():
    if STATE["model"] is None:
        return jsonify({"error": "Please train the model first."}), 400

    y_train = STATE["y_train"]
    y_pred_train = STATE["y_pred_train"]
    target_name = STATE["target_name"] or "Target"
    metrics = STATE["metrics"]

    # --- Plot 1: Actual vs Predicted scatter plot with ideal-fit line ---
    fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
    ax1.scatter(y_train, y_pred_train, color="#4f46e5", edgecolor="white", s=70, label="Data points")
    min_val = min(y_train.min(), y_pred_train.min())
    max_val = max(y_train.max(), y_pred_train.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linewidth=2, label="Ideal fit (y = x)")
    ax1.set_xlabel(f"Actual {target_name}")
    ax1.set_ylabel(f"Predicted {target_name}")
    ax1.set_title("Regression Plot: Actual vs Predicted")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig1.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=110)
    plt.close(fig1)
    buf1.seek(0)
    plot1_b64 = base64.b64encode(buf1.read()).decode("utf-8")

    # --- Plot 2: Metrics bar/scatter chart + residual scatter ---
    residuals = y_train - y_pred_train

    fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
    ax2.scatter(range(len(residuals)), residuals, color="#10b981", edgecolor="white", s=70)
    ax2.axhline(0, color="#ef4444", linewidth=2, linestyle="--")
    ax2.set_xlabel("Training sample index")
    ax2.set_ylabel("Residual (Actual - Predicted)")
    ax2.set_title(
        f"Residuals & Error Metrics\nR2={metrics['r2']:.3f}  MSE={metrics['mse']:.3f}  "
        f"RMSE={metrics['rmse']:.3f}  MAE={metrics['mae']:.3f}"
    )
    ax2.grid(alpha=0.3)
    fig2.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=110)
    plt.close(fig2)
    buf2.seek(0)
    plot2_b64 = base64.b64encode(buf2.read()).decode("utf-8")

    return jsonify({
        "regression_plot": plot1_b64,
        "metrics_plot": plot2_b64,
    })


@app.route("/reset", methods=["POST"])
def reset():
    reset_state()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    app.run(debug=True)
