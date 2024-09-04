import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

# Load data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Display the first few rows of the dataset
print(data.head())

# Total count including duplicates
total_count = len(data)

# Remove duplicates
data_cleaned = data.drop_duplicates()

# Save the cleaned dataset to a new CSV file
data_cleaned.to_csv('cleaned_diabetes_dataset.csv', index=False)

# Count how many diabetic and non-diabetic people there are before removing duplicates
diabetic_count_before = data[data['diabetes'] == 1]['diabetes'].count()
non_diabetic_count_before = data[data['diabetes'] == 0]['diabetes'].count()

# Count how many diabetic and non-diabetic people there are after removing duplicates
diabetic_count_after = data_cleaned[data_cleaned['diabetes'] == 1]['diabetes'].count()
non_diabetic_count_after = data_cleaned[data_cleaned['diabetes'] == 0]['diabetes'].count()

print("Diabetic Count before removing duplicates:", diabetic_count_before)
print("Non-Diabetic Count before removing duplicates:", non_diabetic_count_before)
print("Total Count before removing duplicates:", total_count)

print("Diabetic Count after removing duplicates:", diabetic_count_after)
print("Non-Diabetic Count after removing duplicates:", non_diabetic_count_after)
print("Total Count after removing duplicates:", len(data_cleaned))



# Calculate the ratio of diabetic and non-diabetic people
diabetic_ratio = diabetic_count_after / non_diabetic_count_after

print("Ratio of Diabetic to Non-Diabetic People:", diabetic_ratio)


# Separate diabetic and non-diabetic records
diabetic_data = data[data['diabetes'] == 1]
non_diabetic_data = data[data['diabetes'] == 0]

# Calculate the number of records needed for each class
num_diabetic = int(2000 * 0.3)
num_non_diabetic = int(2000 * 0.6)

# Randomly sample records for each class
sampled_diabetic = diabetic_data.sample(n=num_diabetic, random_state=42)
sampled_non_diabetic = non_diabetic_data.sample(n=num_non_diabetic, random_state=42)

# Concatenate sampled data
sampled_data = pd.concat([sampled_diabetic, sampled_non_diabetic])

# Shuffle the sampled data
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the sampled data to a new CSV file
sampled_data.to_csv('sampled_diabetes_dataset.csv', index=False)


# Load the sampled dataset
sampled_data = pd.read_csv('sampled_diabetes_dataset.csv')

# Count how many diabetic and non-diabetic people are in the sampled dataset
diabetic_count_sampled = sampled_data[sampled_data['diabetes'] == 1]['diabetes'].count()
non_diabetic_count_sampled = sampled_data[sampled_data['diabetes'] == 0]['diabetes'].count()

print("Diabetic Count in the sampled dataset:", diabetic_count_sampled)
print("Non-Diabetic Count in the sampled dataset:", non_diabetic_count_sampled)