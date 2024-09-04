import pandas as pd
from sklearn.model_selection import train_test_split
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
data_cleaned.to_csv('cleaned_diabetes_prediction_dataset.csv', index=False)

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
# dataset with 2000 records
num_diabetic = 194
num_non_diabetic = 1806

# dataset with 1000 records
# num_diabetic = 97
# num_non_diabetic = 903

# Randomly sample records for each class
sampled_diabetic = diabetic_data.sample(n=num_diabetic, random_state=42)
sampled_non_diabetic = non_diabetic_data.sample(n=num_non_diabetic, random_state=42)

# Concatenate sampled data
sampled_data = pd.concat([sampled_diabetic, sampled_non_diabetic])

# Shuffle the sampled data
sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the sampled data to a new CSV file
sampled_data.to_csv('cleaned_diabetes_prediction_dataset.csv', index=False)


# Load the sampled dataset
sampled_data = pd.read_csv('cleaned_diabetes_prediction_dataset.csv')

# Count how many diabetic and non-diabetic people are in the sampled dataset
diabetic_count_sampled = sampled_data[sampled_data['diabetes'] == 1]['diabetes'].count()
non_diabetic_count_sampled = sampled_data[sampled_data['diabetes'] == 0]['diabetes'].count()

print("Diabetic Count in the sampled dataset:", diabetic_count_sampled)
print("Non-Diabetic Count in the sampled dataset:", non_diabetic_count_sampled)


# Load the dataset
data = pd.read_csv('cleaned_diabetes_prediction_dataset.csv')

# Convert relevant columns to numeric
numeric_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values after conversion
data.dropna(subset=numeric_columns, inplace=True)

# Group the data by the 'diabetes' column
grouped_data = data.groupby('diabetes')

# Define a function to compute the average, range, and 90% range of each attribute
def compute_stats(group):
    stats = {}
    stats['count'] = group.shape[0]  # Number of individuals in the group
    stats['average_age'] = group['age'].mean()
    stats['age_range'] = (group['age'].min(), group['age'].max())
    stats['average_bmi'] = group['bmi'].mean()
    stats['bmi_range'] = (group['bmi'].min(), group['bmi'].max())
    stats['average_glucose_level'] = group['blood_glucose_level'].mean()
    stats['glucose_level_range'] = (group['blood_glucose_level'].min(), group['blood_glucose_level'].max())
    stats['average_HbA1c_level'] = group['HbA1c_level'].mean()
    stats['HbA1c_level_range'] = (group['HbA1c_level'].min(), group['HbA1c_level'].max())
    
    # Compute the 5th and 95th percentiles for each attribute
    percentiles = group[numeric_columns].quantile([0.05, 0.95])
    stats['90_percentile_age'] = (percentiles.loc[0.05, 'age'], percentiles.loc[0.95, 'age'])
    stats['90_percentile_bmi'] = (percentiles.loc[0.05, 'bmi'], percentiles.loc[0.95, 'bmi'])
    stats['90_percentile_glucose_level'] = (percentiles.loc[0.05, 'blood_glucose_level'], percentiles.loc[0.95, 'blood_glucose_level'])
    stats['90_percentile_HbA1c_level'] = (percentiles.loc[0.05, 'HbA1c_level'], percentiles.loc[0.95, 'HbA1c_level'])

    return pd.Series(stats, index=['count', 'average_age', 'age_range', 
                                   'average_bmi', 'bmi_range', 
                                   'average_glucose_level', 'glucose_level_range', 
                                   'average_HbA1c_level', 'HbA1c_level_range',
                                   '90_percentile_age', '90_percentile_bmi',
                                   '90_percentile_glucose_level', '90_percentile_HbA1c_level'])

# Compute statistics for diabetic and non-diabetic groups
stats = grouped_data.apply(compute_stats)

# Display the statistics
print("Statistics for Diabetic People:")
print(stats.loc[1])
print("\nStatistics for Non-Diabetic People:")
print(stats.loc[0])

# Load the dataset
df = pd.read_csv('cleaned_diabetes_prediction_dataset.csv')

# Split features (X) and target variable (y)
X = df.drop('diabetes', axis=1)  # Features
y = df['diabetes']                # Target variable

# Split the data into training and test sets (70% train, 30% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create directories if they don't exist
if not os.path.exists('train_set_2'):
    os.makedirs('train_set_2')

if not os.path.exists('test_set_2'):
    os.makedirs('test_set_2')

# Save training set
X_train.to_csv('train_set_2/X_train.csv', index=False)
y_train.to_csv('train_set_2/y_train.csv', index=False)

# Save testing set
X_test.to_csv('test_set_2/X_test.csv', index=False)
y_test.to_csv('test_set_2/y_test.csv', index=False)


original_counts = y.value_counts()
print("Original Dataset:")
print("Non-Diabetic (Outcome 0):", original_counts[0])
print("Diabetic (Outcome 1):", original_counts[1])

# Count the occurrences of each class in the training set
train_counts = y_train.value_counts()
print("\nTraining Set:")
print("Non-Diabetic (Outcome 0):", train_counts[0])
print("Diabetic (Outcome 1):", train_counts[1])

# Count the occurrences of each class in the test set
test_counts = y_test.value_counts()
print("\nTest Set:")
print("Non-Diabetic (Outcome 0):", test_counts[0])
print("Diabetic (Outcome 1):", test_counts[1])


