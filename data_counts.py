import pandas as pd

# Load data
df = pd.read_csv('train.csv')

# Count the number of instances for each label
class_counts = df['Class'].value_counts()

print('missing values')
# Check if there are any missing values in the dataframe
missing_values = df.isnull().sum()

# Print the number of missing values for each column
print(missing_values)

print(class_counts)
print('in file \'greeks.csv\'')
# Load data
df = pd.read_csv('greeks.csv')

# List of labels to count
labels = ['A', 'B', 'D', 'G']

# Loop through each label and count the occurrences
for label in labels:
    count = (df['Alpha'] == label).sum()
    print(f"The label '{label}' occurs {count} times.")
