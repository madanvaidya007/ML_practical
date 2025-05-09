# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Step 2: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Step 4: Create the Naive Bayes classifier and train it
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Step 5: Predict on the test data
y_pred = gnb.predict(X_test)

# Step 6: Evaluate the accuracy
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print("✅ Gaussian Naive Bayes Model Accuracy (in %): {:.2f}".format(accuracy))
