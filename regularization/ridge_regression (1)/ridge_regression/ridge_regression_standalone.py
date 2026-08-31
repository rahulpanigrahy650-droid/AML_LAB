import numpy as np
import matplotlib.pyplot as plt


class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, alpha=0.1):
        self.bias = None
        self.weights = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha       # regularization strength
        self.cost_history = []   # stores cost (MSE + L2 penalty) at every iteration

    def fit(self, x, y):  # x_train, y_train
        m, n = x.shape  # (number of samples, number of features)

        # step 1 - initialize params
        self.bias = 0
        self.weights = np.zeros(n)

        # Gradient descent
        for i in range(self.n_iter):

            # step 2 - calc y_pred
            y_pred = np.dot(x, self.weights) + self.bias

            # step 3 - calc gradient (MSE part + L2 penalty on weights)
            db = (1 / m) * np.sum(y_pred - y)
            dw = (1 / m) * np.dot(x.T, (y_pred - y)) + 2 * self.alpha * self.weights

            # step 4 - convergence theorem - params update
            self.bias -= self.lr * db
            self.weights -= self.lr * dw

            # step 5 - track cost for this iteration (MSE + L2 penalty)
            mse = (1 / m) * np.sum((y_pred - y) ** 2)
            cost = mse + self.alpha * np.sum(self.weights ** 2)
            self.cost_history.append(cost)

    def predict(self, x):
        y_pred = self.bias + np.dot(x, self.weights)
        return y_pred


def calculate_metrics(y, y_pred):
    mse = np.mean((y - y_pred) ** 2)
    mae = np.mean(np.abs(y - y_pred))
    rmse = np.sqrt(mse)

    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    r2 = 0 if ss_total == 0 else 1 - (ss_residual / ss_total)

    return mse, mae, rmse, r2


if __name__ == "__main__":

    # -------- Dataset --------
    x = np.array([[1], [2], [3], [4]])
    y = np.array([2, 2, 5, 4])

    # -------- Train model --------
    model = RidgeRegression(learning_rate=0.01, n_iter=1000, alpha=0.1)
    model.fit(x, y)

    y_pred = model.predict(x)
    mse, mae, rmse, r2 = calculate_metrics(y, y_pred)

    print("Final Weight(s):", model.weights)
    print("Final Bias:", model.bias)
    print(f"Equation: y = {model.weights[0]:.4f}x + {model.bias:.4f}")
    print(f"MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    # -------- Graph 1: Regression Fit --------
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, label="Original Data")

    sort_index = np.argsort(x[:, 0])
    plt.plot(x[sort_index, 0], y_pred[sort_index], color="red", label="Ridge Regression Line")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ridge Regression with Gradient Descent")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ridge_regression_fit.png")
    plt.show()

    # -------- Graph 2: Cost History --------
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, model.n_iter + 1), model.cost_history, color="steelblue")

    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost (MSE + L2 Regularization)")
    plt.title("Cost History during Gradient Descent")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("ridge_regression_cost.png")
    plt.show()
