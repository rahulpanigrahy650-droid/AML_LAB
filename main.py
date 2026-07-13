import numpy as np
import matplotlib.pyplot as plt

# Input
x = list(map(float, input("Enter x values (comma-separated): ").split(",")))
y = list(map(float, input("Enter y values (comma-separated): ").split(",")))

n = len(x)

# Calculate required sums
sum_x = sum(x)
sum_y = sum(y)
sum_xy = 0
sum_x2 = 0

for i in range(n):
    sum_xy += x[i] * y[i]
    sum_x2 += x[i] * x[i]

# Calculate slope (m) and intercept (c)
slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
intercept = (sum_y - slope * sum_x) / n

# Predict y values
y_pred = []

for value in x:
    y_pred.append(slope * value + intercept)

# Calculate MAE and MSE
mae = 0
mse = 0

for i in range(n):
    error = y[i] - y_pred[i]
    mae += abs(error)
    mse += error ** 2

mae = mae / n
mse = mse / n
rmse = mse ** 0.5

# Calculate R² Score
y_mean = sum_y / n

ss_res = 0
ss_tot = 0

for i in range(n):
    ss_res += (y[i] - y_pred[i]) ** 2
    ss_tot += (y[i] - y_mean) ** 2

r2 = 1 - (ss_res / ss_tot)

# Print results
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

# Graph 2: Model Metrics
metrics = ["MSE", "MAE", "RMSE", "R2"]
values = [mse, mae, rmse, r2]

plt.figure(figsize=(6, 4))
plt.scatter(metrics, values, color="green")

for i in range(len(metrics)):
    plt.text(metrics[i], values[i], round(values[i], 3))

plt.xlabel("Metrics")
plt.ylabel("Value")
plt.title("Evaluation Metrics")
plt.grid(True)
plt.show()