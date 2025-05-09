import csv

num_attributes = 6
a = []

print("\nThe Given Training Data Set:\n")
with open('Weather.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        a.append(row)
        print(row)

# Initialize the hypothesis with the most specific values
hypothesis = ['0'] * num_attributes
print("\nThe initial value of hypothesis:\n", hypothesis)

# Find-S Algorithm
print("\nFind-S: Finding a Maximally Specific Hypothesis\n")

# Set the hypothesis to the first positive example
for j in range(num_attributes):
    hypothesis[j] = a[0][j]

# Go through the training examples
for i in range(1, len(a)):
    if a[i][num_attributes].lower() == 'yes':
        for j in range(num_attributes):
            if a[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
    print(f"For Training instance No:{i} the hypothesis is: {hypothesis}")

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypothesis)
