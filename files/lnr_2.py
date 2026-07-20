import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Target Variable 
target = input("Enter Target Variable Name: ")

data = {}
data[target] = list(map(float, input(f"Enter {target} values (comma-separated): ").split(",")))

#  Input Features 
n = int(input("Enter Number of Input Features: "))

feature_names = []

for i in range(n):
    name = input(f"\nEnter Feature {i+1} Name: ")
    feature_names.append(name)

    values = list(map(float, input(f"Enter {name} values (comma-separated): ").split(",")))
    data[name] = values

# Create DataFrame 
cols = feature_names + [target]
df = pd.DataFrame(data)[cols]

print("\nDataFrame")
print(df)

#  X and y 
X = df[feature_names]
y = df[target]

#Train Model -
model = LinearRegression()
model.fit(X, y)

#  Prediction on Training Data
y_pred = model.predict(X)

#  Metrics 
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

#- Equation 
print("\nRegression Equation:\n")

equation = f"{target} = {model.intercept_:.2f}"

for i in range(len(feature_names)):
    equation += f" + ({model.coef_[i]:.2f} × {feature_names[i]})"

print(equation)

#  Test New Data 
print("\nEnter Values to Predict")

test = []

for feature in feature_names:
    value = float(input(f"{feature}: "))
    test.append(value)

prediction = model.predict([test])

print(f"\nPredicted {target}: {prediction[0]:.2f}")

# Metrics
print("\nEvaluation Metrics")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# Regression Graph

plt.figure(figsize=(6,4))
plt.scatter(y, y_pred, color="blue", label="Data")
plt.plot([min(y), max(y)], [min(y), max(y)], color="red", label="Best Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Multiple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Metrics Scatter Plot
plt.figure(figsize=(6,4))

metric_names = ["MAE", "MSE", "RMSE", "R²"]
metric_values = [mae, mse, rmse, r2]

plt.scatter(metric_names, metric_values, s=120)

for i in range(len(metric_names)):
    plt.text(metric_names[i], metric_values[i], round(metric_values[i], 2))

plt.title("Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")
plt.grid(True)
plt.show()