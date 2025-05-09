# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# Step 2: Load the dataset
df = pd.read_csv('data.csv')  # Replace path as needed
print(df.head(5))  # Preview first 5 rows

# Step 3: Visualize Home Wins vs Losses
sns.countplot(x='WINorLOSS', hue='Home', data=df)
plt.title("Home vs Away Wins")
plt.xlabel("Match Result")
plt.ylabel("Number of Matches")
plt.show()

# Step 4: Import required modules for model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 5: Prepare data for training
X = df.drop(['Home'], axis=1)  # Features (drop target column)
y = df['Home']  # Target column (Home team win/loss)

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=1
)

# Step 7: Train the logistic regression model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Step 8: Make predictions and evaluate
predictions = logmodel.predict(X_test)

# Step 9: Output results
print("\n✅ Classification Report:\n", classification_report(y_test, predictions))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("🎯 Accuracy Score: {:.2f}%".format(accuracy_score(y_test, predictions) * 100))
