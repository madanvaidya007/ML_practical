# Step 1: Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Step 2: Load the Iris dataset
iris = load_iris()

# Display dataset information
print("🔸 Feature Names:", iris.feature_names)
print("🔸 Target Names:", iris.target_names)
print("🔸 Sample Data (first 5 rows):\n", iris.data[:5])
print("🔸 Target Labels (first 5):", iris.target[:5])

# Step 3: Split the data into training and test sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=0
)

# Step 4: Create the KNN classifier model
knn = KNeighborsClassifier(n_neighbors=5)

# Step 5: Train the model using the training data
knn.fit(X_train, y_train)

# Step 6: Test the accuracy of the model
accuracy = knn.score(X_test, y_test)
print("\n✅ Accuracy of the KNN model: {:.2f}".format(accuracy))

# Step 7: Predict using the test dataset
predictions = knn.predict(X_test)

print("\n🔹 Predicted Labels:\n", predictions)
print("🔹 Actual Labels:\n", y_test)

# Step 8: Identify misclassifications
misclassified_count = (predictions != y_test).sum()
print("\n❌ Total Misclassified Samples:", misclassified_count)

# Optional: Display misclassified indices
print("\n🔍 Misclassified Data Points (Index: Predicted ≠ Actual):")
for i in range(len(y_test)):
    if predictions[i] != y_test[i]:
        print(f"  Index {i}: Predicted = {predictions[i]}, Actual = {y_test[i]}")
