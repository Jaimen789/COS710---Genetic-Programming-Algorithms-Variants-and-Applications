import pandas as pd

# Load the dataset
data = pd.read_csv('sampled_diabetes_dataset.csv')

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
