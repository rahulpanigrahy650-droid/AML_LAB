import numpy as np
import matplotlib.pyplot as plt

# Input
x = np.array(list(map(float, input("Enter x values (comma-separated): ").split(","))))
y = np.array(list(map(float, input("Enter y values (comma-separated): ").split(","))))

n = len(x)

# Calculate sums using NumPy
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x ** 2)

# Calculate slope and intercept
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
intercept = (sum_y - slope * sum_x) / n

# Predict y values
y_pred = slope * x + intercept

# Evaluation Metrics
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)

# R² Score
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# Print Results
print("\nSlope (m):", round(slope, 4))
print("Intercept (c):", round(intercept, 4))
print("Equation: y =", round(slope, 4), "x +", round(intercept, 4))

print("\nEvaluation Metrics")
print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R²  :", round(r2, 4))

# Graph 1: Regression Line
plt.figure(figsize=(6, 4))
plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Graph 2: Metrics
metrics = ["MSE", "MAE", "RMSE", "R2"]
values = [mse, mae, rmse, r2]

plt.figure(figsize=(6, 4))
plt.scatter(metrics, values, color="green")

for i in range(len(metrics)):
    plt.text(metrics[i], values[i], round(values[i], 4))

plt.xlabel("Metrics")
plt.ylabel("Value")
plt.title("Evaluation Metrics")
plt.grid(True)
plt.show()