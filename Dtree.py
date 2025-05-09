import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Load dataset
df = pd.read_csv("salaries.csv")
print("\nDataset Preview:\n", df.head())

# Split data into features and target
inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

# Encode categorical variables
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

# Final input for the model
inputs_n = inputs.drop(['company', 'job', 'degree'], axis='columns')

# Build and train decision tree model
model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

# Accuracy score
print("\nModel Accuracy:", model.score(inputs_n, target))

# Prediction examples
print("\nPrediction: Google, Computer Programmer, Bachelors")
print("Predicted:", model.predict([[2, 1, 0]]))  # Should print [0]

print("\nPrediction: Google, Computer Programmer, Masters")
print("Predicted:", model.predict([[2, 1, 1]]))  # Should print [1]
