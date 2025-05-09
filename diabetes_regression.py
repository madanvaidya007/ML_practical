import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature (BMI - feature at index 2)
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create linear regression object
model = linear_model.LinearRegression()

# Train the model using the training sets
model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = model.predict(diabetes_X_test)

# Print evaluation metrics
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print("Variance score (R²): %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.figure(figsize=(8, 5))
plt.scatter(diabetes_X_test, diabetes_y_test, color='black', label='Actual data')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=2, label='Predicted regression line')

# Add labels and title
plt.xlabel('BMI (Body Mass Index)')
plt.ylabel('Disease Progression')
plt.title('Linear Regression on Diabetes Dataset (BMI vs Progression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
